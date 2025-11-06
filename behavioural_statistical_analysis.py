#!/usr/bin/env python3
"""
Statistical Analysis for BCI Confirmation Bias Study
================================================================================================
Integrates ALL questionnaire data including:
- BIS/BAS personality measures (24 items)
- Topic-specific beliefs and familiarity (12 topics × 2 = 24 items)
- Decision-making style measures (5 items)
- Demographics (age)

Uses CORRECT Monte Carlo baseline (20.8%) for hypothesis testing
Tests multiple hypotheses about confirmation bias patterns

Author: Jason Stewart
Ethics: ETH23-7909
Updated: November 2025
================================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveBCIStatisticalAnalysis:
    """
    Complete statistical analysis with ALL questionnaire variables
    
    Hypotheses Tested:
    ------------------
    H0: No systematic confirmation bias (μ_CB = μ_chance ≈ 20%)
    
    H1: Confirmation bias prevalence
        - CB rate significantly exceeds chance level
        
    H2: BIS/BAS personality effects
        - H2a: Higher BIS -> Greater confirmation bias
        - H2b: Higher BAS -> Greater disconfirmation seeking
        
    H3: Topic-specific belief effects
        - H3a: Higher belief extremity -> Greater confirmation bias
        - H3b: Higher familiarity -> Stronger bias patterns
        
    H4: Decision-making style effects
        - H4a: Confidence level -> Confirmation bias
        - H4b: Response to challenge -> Information seeking
        - H4c: Mind-changing frequency -> Cognitive flexibility
        
    H5: Response time patterns
        - H5a: Articles take longer than statements
        - H5b: Disconfirmatory articles take longer than confirmatory
        - H5c: Belief extremity -> Longer RT for challenging info
    """
    
    def __init__(self, 
                 questionnaire_file: str = 'tests_questionnaire_responses.xlsx',
                 behavioral_file: str = './bci_output/patched/batch_behavioural/batch_summary.csv',
                 monte_carlo_baseline: float = 0.208,
                 output_dir: str = './bci_output/patched/batch_behavioural/statistical_analysis'):
        
        self.questionnaire_file = questionnaire_file
        self.behavioral_file = behavioral_file
        self.CORRECT_BASELINE = monte_carlo_baseline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.questionnaire_data = []
        self.behavioral_data = None
        self.merged_data = None
        
        # Topic code mapping (must match experimental design)
        self.TOPIC_MAP = {
            'T01': 'climate',
            'T02': 'tech', 
            'T03': 'economic',
            'T04': 'health',
            'T05': 'education',
            'T06': 'ai',
            'T07': 'work',
            'T15': 'media',
            'T16': 'science',
            'T17': 'parenting',
            'T18': 'aging',
            'T20': 'mental'
        }
        
    def load_complete_questionnaire_data(self):
        """
        Load ALL 56 questionnaire columns including:
        - BIS/BAS (cols 1-24)
        - Topic beliefs + familiarity (cols 25-48)  
        - Decision-making style (cols 49-53)
        - Demographics (col 55)
        """
        print(f"Loading COMPLETE questionnaire data from {self.questionnaire_file}")
        print("="*70)
        
        df = pd.read_excel(self.questionnaire_file, header=0)
        
        for index, row in df.iterrows():
            participant_id = str(row.iloc[0])
            if 'participant_' in participant_id:
                participant_id = participant_id.replace('participant_', '')
            
            # === 1. BIS/BAS Scores (existing, verified correct) ===
            bis_bas_responses = row.iloc[1:25].values.tolist()
            bis_bas_scores = self.calculate_bis_bas_scores(participant_id, bis_bas_responses)
            
            # === 2. Topic Beliefs and Familiarity (NEW) ===
            # Columns 25-48: 12 topics × (belief + familiarity)
            topic_beliefs = {}
            for i, (topic_code, topic_name) in enumerate(self.TOPIC_MAP.items()):
                belief_col = 25 + (i * 2)      # 25, 27, 29, ...
                familiarity_col = 26 + (i * 2)  # 26, 28, 30, ...
                
                topic_beliefs[f'{topic_code}_belief'] = int(row.iloc[belief_col]) \
                    if pd.notna(row.iloc[belief_col]) else None
                topic_beliefs[f'{topic_code}_familiarity'] = int(row.iloc[familiarity_col]) \
                    if pd.notna(row.iloc[familiarity_col]) else None
            
            # Calculate belief extremity scores (distance from neutral midpoint = 3)
            extremity_scores = {}
            for topic_code in self.TOPIC_MAP.keys():
                belief_key = f'{topic_code}_belief'
                if topic_beliefs.get(belief_key) is not None:
                    extremity_scores[f'{topic_code}_extremity'] = abs(topic_beliefs[belief_key] - 3)
            
            # Average extremity across all topics
            valid_extremities = [v for v in extremity_scores.values() if v is not None]
            extremity_scores['mean_belief_extremity'] = np.mean(valid_extremities) \
                if valid_extremities else None
            
            # === 3. Decision-Making Style (NEW) ===
            decision_style = {
                'confidence_level': str(row.iloc[49]) if pd.notna(row.iloc[49]) else None,
                'response_to_challenge': str(row.iloc[50]) if pd.notna(row.iloc[50]) else None,
                'mind_changing_frequency': str(row.iloc[51]) if pd.notna(row.iloc[51]) else None,
                'decision_basis': str(row.iloc[52]) if pd.notna(row.iloc[52]) else None,
                'others_influence': str(row.iloc[53]) if pd.notna(row.iloc[53]) else None,
            }
            
            # Encode categorical variables as numeric for analysis
            confidence_map = {
                '1 - Not at all confident': 1, '2 - Slightly confident': 2,
                '3 - Moderately confident': 3, '4 - Very confident': 4,
                '5 - Extremely confident': 5,
                'Not at all confident': 1, 'Slightly confident': 2,
                'Moderately confident': 3, 'Very confident': 4,
                'Extremely confident': 5
            }
            
            challenge_map = {
                'Immediately dismiss it': 1,
                'Feel defensive': 2, 
                'Consider it carefully': 4,
                'Actively seek more information': 5
            }
            
            mind_change_map = {
                '1 - Rarely': 1, '2 - Rarely': 1,
                '2 - Occasionally': 2, '3 - Sometimes': 3,
                '4 - Frequently': 4, '5 - Very frequently': 5,
                'Rarely': 1, 'Occasionally': 2, 'Sometimes': 3,
                'Frequently': 4, 'Very frequently': 5
            }
            
            decision_style['confidence_numeric'] = confidence_map.get(
                decision_style['confidence_level'], None)
            decision_style['challenge_openness'] = challenge_map.get(
                decision_style['response_to_challenge'], None)
            decision_style['cognitive_flexibility'] = mind_change_map.get(
                decision_style['mind_changing_frequency'], None)
            
            # === 4. Demographics (NEW) ===
            demographics = {
                'age_range': str(row.iloc[55]) if pd.notna(row.iloc[55]) else None,
            }
            
            # === Merge all data ===
            complete_record = {
                **bis_bas_scores,
                **topic_beliefs,
                **extremity_scores,
                **decision_style,
                **demographics
            }
            
            self.questionnaire_data.append(complete_record)
            
            print(f"  P{participant_id}:")
            print(f"    BIS={complete_record['bis_score']:.2f}, "
                  f"BAS={complete_record['bas_total']:.2f}")
            print(f"    Mean belief extremity={complete_record.get('mean_belief_extremity', 0):.2f}")
            print(f"    Confidence={complete_record.get('confidence_level', 'N/A')}")
            print(f"    Cognitive flexibility={complete_record.get('cognitive_flexibility', 'N/A')}")
        
        print(f"\n[OK] Loaded data for {len(self.questionnaire_data)} participants")
        return pd.DataFrame(self.questionnaire_data)
    
    def calculate_bis_bas_scores(self, participant_id: str, responses: list):
        """
        Calculate BIS/BAS subscale scores with proper reversals
        
        BIS (Behavioral Inhibition System): 7 items
        BAS (Behavioral Activation System): 13 items
          - Drive: 4 items
          - Fun Seeking: 4 items  
          - Reward Responsiveness: 5 items
        """
        responses = [int(r) if str(r).isdigit() else r for r in responses]
        
        # BIS items (indices in 0-based array)
        bis_items = [
            6 - responses[1],   # Q2 reversed
            responses[7],       # Q8
            responses[12],      # Q13
            responses[15],      # Q16
            responses[18],      # Q19
            6 - responses[21],  # Q22 reversed
            responses[23]       # Q24
        ]
        
        # BAS subscales
        bas_drive = [responses[i] for i in [2, 8, 11, 20]]
        bas_fun = [responses[i] for i in [4, 9, 14, 19]]
        bas_reward = [responses[i] for i in [3, 6, 13, 17, 22]]
        
        return {
            'participant_id': participant_id,
            'bis_score': np.mean(bis_items),
            'bas_drive': np.mean(bas_drive),
            'bas_fun_seeking': np.mean(bas_fun),
            'bas_reward': np.mean(bas_reward),
            'bas_total': np.mean(bas_drive + bas_fun + bas_reward)
        }
    
    def load_behavioral_data(self):
        """Load behavioral summary from batch processing"""
        print(f"\nLoading behavioral data from {self.behavioral_file}")
        self.behavioral_data = pd.read_csv(self.behavioral_file)
        print(f"  Found data for {len(self.behavioral_data)} participants")
        
        # Display key metrics
        print("\nBehavioral metrics summary:")
        print(f"  Mean CB rate: {self.behavioral_data['confirmation_bias_rate'].mean():.1%}")
        print(f"  Mean disconf rate: {self.behavioral_data['disconfirmation_rate'].mean():.1%}")
        print(f"  Mean statement RT: {self.behavioral_data['mean_statement_rt'].mean():.1f}s")
        print(f"  Mean article RT: {self.behavioral_data['mean_article_rt'].mean():.1f}s")
        
        return self.behavioral_data
    
    def test_h0_baseline(self):
        """
        H0: No systematic confirmation bias beyond chance
        
        Test: One-sample t-test comparing observed CB rates to Monte Carlo baseline
        """
        print("\n" + "="*70)
        print("NULL HYPOTHESIS TESTING (H0)")
        print("="*70)
        
        cb_rates = self.merged_data['confirmation_bias_rate'].values
        n = len(cb_rates)
        
        print(f"\nH0: μ_CB = {self.CORRECT_BASELINE:.1%} (chance level from Monte Carlo)")
        print(f"H1: μ_CB > {self.CORRECT_BASELINE:.1%} (systematic confirmation bias)")
        
        # One-sample t-test
        t_stat, p_value_two_tailed = stats.ttest_1samp(cb_rates, self.CORRECT_BASELINE)
        p_value_one_tailed = p_value_two_tailed / 2 if t_stat > 0 else 1 - (p_value_two_tailed / 2)
        
        # Effect size
        mean_cb = np.mean(cb_rates)
        std_cb = np.std(cb_rates, ddof=1)
        cohens_d = (mean_cb - self.CORRECT_BASELINE) / std_cb
        
        # Individual z-scores
        print(f"\nObserved data:")
        print(f"  Mean CB rate: {mean_cb:.1%} (SD={std_cb:.1%})")
        print(f"  Range: {np.min(cb_rates):.1%} to {np.max(cb_rates):.1%}")
        print(f"  Ratio to chance: {mean_cb/self.CORRECT_BASELINE:.2f}x")
        
        print(f"\nStatistical test results:")
        print(f"  t({n-1}) = {t_stat:.3f}")
        print(f"  p-value (one-tailed) = {p_value_one_tailed:.4f}")
        print(f"  p-value (two-tailed) = {p_value_two_tailed:.4f}")
        print(f"  Cohen's d = {cohens_d:.3f}")
        
        # Interpretation
        alpha = 0.05
        if p_value_one_tailed < alpha:
            print(f"\n  [OK] REJECT H0 (p < {alpha})")
            print(f"  -> Confirmation bias significantly ABOVE chance")
            print(f"  -> Effect size: {self._interpret_cohens_d(cohens_d)}")
        else:
            print(f"\n  ✗ FAIL TO REJECT H0 (p ≥ {alpha})")
            print(f"  -> Insufficient evidence for systematic bias")
            print(f"  -> Note: Small sample size (N={n}) limits statistical power")
        
        return {
            'mean_cb': mean_cb,
            'baseline': self.CORRECT_BASELINE,
            'difference': mean_cb - self.CORRECT_BASELINE,
            'ratio': mean_cb / self.CORRECT_BASELINE,
            't_stat': t_stat,
            'p_one_tailed': p_value_one_tailed,
            'p_two_tailed': p_value_two_tailed,
            'cohens_d': cohens_d,
            'reject_h0': p_value_one_tailed < alpha,
            'effect_interpretation': self._interpret_cohens_d(cohens_d)
        }
    
    def test_h2_bis_bas_effects(self):
        """
        H2: BIS/BAS personality effects on confirmation bias
        
        H2a: Higher BIS -> Greater confirmation bias (threat avoidance)
        H2b: Higher BAS -> Greater disconfirmation seeking (reward seeking)
        """
        print("\n" + "="*70)
        print("HYPOTHESIS 2: BIS/BAS PERSONALITY EFFECTS")
        print("="*70)
        
        results = {}
        
        # H2a: BIS -> Confirmation Bias
        print("\nH2a: BIS -> Confirmation Bias")
        print("-" * 50)
        
        bis_cb_r, bis_cb_p = stats.pearsonr(
            self.merged_data['bis_score'],
            self.merged_data['confirmation_bias_rate']
        )
        
        print(f"  Pearson r = {bis_cb_r:.3f}, p = {bis_cb_p:.3f}")
        
        if bis_cb_p < 0.05:
            direction = "positive" if bis_cb_r > 0 else "negative"
            print(f"  [OK] SIGNIFICANT {direction} correlation")
            if bis_cb_r > 0:
                print(f"  -> Higher BIS -> More confirmation bias (supports theory)")
            else:
                print(f"  -> Higher BIS -> Less confirmation bias (contradicts theory)")
        else:
            print(f"  - Not significant (p ≥ 0.05)")
        
        results['h2a'] = {
            'correlation': bis_cb_r,
            'p_value': bis_cb_p,
            'significant': bis_cb_p < 0.05,
            'supports_theory': bis_cb_r > 0 and bis_cb_p < 0.05
        }
        
        # H2b: BAS -> Disconfirmation Seeking
        print("\nH2b: BAS -> Disconfirmation Seeking")
        print("-" * 50)
        
        bas_ds_r, bas_ds_p = stats.pearsonr(
            self.merged_data['bas_total'],
            self.merged_data['disconfirmation_rate']
        )
        
        print(f"  Pearson r = {bas_ds_r:.3f}, p = {bas_ds_p:.3f}")
        
        if bas_ds_p < 0.05:
            direction = "positive" if bas_ds_r > 0 else "negative"
            print(f"  [OK] SIGNIFICANT {direction} correlation")
            if bas_ds_r > 0:
                print(f"  -> Higher BAS -> More disconfirmation (supports theory)")
            else:
                print(f"  -> Higher BAS -> Less disconfirmation (PARADOX!)")
        else:
            print(f"  - Not significant (p ≥ 0.05)")
        
        results['h2b'] = {
            'correlation': bas_ds_r,
            'p_value': bas_ds_p,
            'significant': bas_ds_p < 0.05,
            'supports_theory': bas_ds_r > 0 and bas_ds_p < 0.05
        }
        
        # Additional BAS subscale analyses
        print("\nBAS Subscale Breakdown:")
        for subscale in ['bas_drive', 'bas_fun_seeking', 'bas_reward']:
            r, p = stats.pearsonr(
                self.merged_data[subscale],
                self.merged_data['disconfirmation_rate']
            )
            print(f"  {subscale}: r = {r:.3f}, p = {p:.3f}")
        
        return results
    
    def test_h3_belief_effects(self):
        """
        H3: Topic-specific belief effects
        
        H3a: Higher belief extremity -> Greater confirmation bias
        H3b: Higher familiarity -> Stronger bias patterns
        """
        print("\n" + "="*70)
        print("HYPOTHESIS 3: TOPIC BELIEF EFFECTS")
        print("="*70)
        
        results = {}
        
        # H3a: Belief Extremity -> Confirmation Bias
        print("\nH3a: Belief Extremity -> Confirmation Bias")
        print("-" * 50)
        
        if 'mean_belief_extremity' in self.merged_data.columns:
            extremity_cb_r, extremity_cb_p = stats.pearsonr(
                self.merged_data['mean_belief_extremity'].dropna(),
                self.merged_data.loc[self.merged_data['mean_belief_extremity'].notna(), 
                                    'confirmation_bias_rate']
            )
            
            print(f"  Pearson r = {extremity_cb_r:.3f}, p = {extremity_cb_p:.3f}")
            
            if extremity_cb_p < 0.05:
                print(f"  [OK] SIGNIFICANT: Stronger beliefs -> More bias")
            else:
                print(f"  - Not significant (may need larger N)")
            
            results['h3a'] = {
                'correlation': extremity_cb_r,
                'p_value': extremity_cb_p,
                'significant': extremity_cb_p < 0.05
            }
        else:
            print("  ⚠ Mean belief extremity not available")
            results['h3a'] = {'error': 'Data not available'}
        
        # H3b: Familiarity effects (aggregated)
        print("\nH3b: Topic Familiarity -> Bias Strength")
        print("-" * 50)
        
        # Calculate mean familiarity if available
        familiarity_cols = [f'{t}_familiarity' for t in self.TOPIC_MAP.keys()]
        available_fam_cols = [col for col in familiarity_cols 
                             if col in self.merged_data.columns]
        
        if available_fam_cols:
            self.merged_data['mean_familiarity'] = self.merged_data[
                available_fam_cols].mean(axis=1)
            
            fam_cb_r, fam_cb_p = stats.pearsonr(
                self.merged_data['mean_familiarity'].dropna(),
                self.merged_data.loc[self.merged_data['mean_familiarity'].notna(),
                                   'confirmation_bias_rate']
            )
            
            print(f"  Pearson r = {fam_cb_r:.3f}, p = {fam_cb_p:.3f}")
            
            if fam_cb_p < 0.05:
                direction = "positive" if fam_cb_r > 0 else "negative"
                print(f"  [OK] SIGNIFICANT {direction} correlation")
                if fam_cb_r > 0:
                    print(f"  -> More familiarity -> Stronger bias (expertise bias)")
                else:
                    print(f"  -> More familiarity -> Less bias (expertise corrects)")
            else:
                print(f"  - Not significant")
            
            results['h3b'] = {
                'correlation': fam_cb_r,
                'p_value': fam_cb_p,
                'significant': fam_cb_p < 0.05
            }
        else:
            print("  ⚠ Familiarity data not available")
            results['h3b'] = {'error': 'Data not available'}
        
        return results
    
    def test_h4_decision_style_effects(self):
        """
        H4: Decision-making style effects
        
        H4a: Confidence level -> Confirmation bias
        H4b: Response to challenge -> Information seeking
        H4c: Mind-changing frequency -> Cognitive flexibility
        """
        print("\n" + "="*70)
        print("HYPOTHESIS 4: DECISION-MAKING STYLE EFFECTS")
        print("="*70)
        
        results = {}
        
        # H4a: Confidence -> Confirmation Bias
        print("\nH4a: Confidence Level -> Confirmation Bias")
        print("-" * 50)
        
        if 'confidence_numeric' in self.merged_data.columns:
            conf_data = self.merged_data[
                self.merged_data['confidence_numeric'].notna()
            ]
            
            if len(conf_data) > 2:
                conf_cb_r, conf_cb_p = stats.pearsonr(
                    conf_data['confidence_numeric'],
                    conf_data['confirmation_bias_rate']
                )
                
                print(f"  Pearson r = {conf_cb_r:.3f}, p = {conf_cb_p:.3f}")
                
                if conf_cb_p < 0.05:
                    if conf_cb_r > 0:
                        print(f"  [OK] SIGNIFICANT: More confident -> More bias")
                    else:
                        print(f"  [OK] SIGNIFICANT: More confident -> Less bias")
                else:
                    print(f"  - Not significant")
                
                results['h4a'] = {
                    'correlation': conf_cb_r,
                    'p_value': conf_cb_p,
                    'significant': conf_cb_p < 0.05
                }
            else:
                print("  ⚠ Insufficient data (N < 3)")
                results['h4a'] = {'error': 'Insufficient data'}
        else:
            print("  ⚠ Confidence data not available")
            results['h4a'] = {'error': 'Data not available'}
        
        # H4b: Challenge Openness -> Disconfirmation
        print("\nH4b: Response to Challenge -> Disconfirmation Seeking")
        print("-" * 50)
        
        if 'challenge_openness' in self.merged_data.columns:
            challenge_data = self.merged_data[
                self.merged_data['challenge_openness'].notna()
            ]
            
            if len(challenge_data) > 2:
                chal_ds_r, chal_ds_p = stats.pearsonr(
                    challenge_data['challenge_openness'],
                    challenge_data['disconfirmation_rate']
                )
                
                print(f"  Pearson r = {chal_ds_r:.3f}, p = {chal_ds_p:.3f}")
                
                if chal_ds_p < 0.05:
                    if chal_ds_r > 0:
                        print(f"  [OK] SIGNIFICANT: More open to challenge -> More disconfirmation")
                    else:
                        print(f"  [OK] SIGNIFICANT: More open to challenge -> Less disconfirmation")
                else:
                    print(f"  - Not significant")
                
                results['h4b'] = {
                    'correlation': chal_ds_r,
                    'p_value': chal_ds_p,
                    'significant': chal_ds_p < 0.05
                }
            else:
                print("  ⚠ Insufficient data (N < 3)")
                results['h4b'] = {'error': 'Insufficient data'}
        else:
            print("  ⚠ Challenge openness data not available")
            results['h4b'] = {'error': 'Data not available'}
        
        # H4c: Cognitive Flexibility -> Bias
        print("\nH4c: Mind-Changing Frequency -> Cognitive Flexibility")
        print("-" * 50)
        
        if 'cognitive_flexibility' in self.merged_data.columns:
            flex_data = self.merged_data[
                self.merged_data['cognitive_flexibility'].notna()
            ]
            
            if len(flex_data) > 2:
                flex_cb_r, flex_cb_p = stats.pearsonr(
                    flex_data['cognitive_flexibility'],
                    flex_data['confirmation_bias_rate']
                )
                
                print(f"  Pearson r = {flex_cb_r:.3f}, p = {flex_cb_p:.3f}")
                
                if flex_cb_p < 0.05:
                    if flex_cb_r < 0:  # Negative correlation expected
                        print(f"  [OK] SIGNIFICANT: More flexible -> Less bias (supports theory)")
                    else:
                        print(f"  ✗ UNEXPECTED: More flexible -> More bias")
                else:
                    print(f"  - Not significant")
                
                results['h4c'] = {
                    'correlation': flex_cb_r,
                    'p_value': flex_cb_p,
                    'significant': flex_cb_p < 0.05,
                    'supports_theory': flex_cb_r < 0 and flex_cb_p < 0.05
                }
            else:
                print("  ⚠ Insufficient data (N < 3)")
                results['h4c'] = {'error': 'Insufficient data'}
        else:
            print("  ⚠ Cognitive flexibility data not available")
            results['h4c'] = {'error': 'Data not available'}
        
        return results
    
    def test_h5_response_time_patterns(self):
        """
        H5: Response time patterns
        
        H5a: Articles take longer than statements
        H5b: Disconfirmatory articles take longer (if RT data available per article type)
        H5c: Belief extremity correlates with RT differences
        """
        print("\n" + "="*70)
        print("HYPOTHESIS 5: RESPONSE TIME PATTERNS")
        print("="*70)
        
        results = {}
        
        # H5a: Statement vs Article RTs
        print("\nH5a: Article RT > Statement RT")
        print("-" * 50)
        
        stmt_rts = self.merged_data['mean_statement_rt'].values
        art_rts = self.merged_data['mean_article_rt'].values
        
        t_stat, p_value = stats.ttest_rel(stmt_rts, art_rts)
        cohens_d = (np.mean(art_rts) - np.mean(stmt_rts)) / np.std(art_rts - stmt_rts)
        
        print(f"  Statement RT: M={np.mean(stmt_rts):.1f}s (SD={np.std(stmt_rts):.1f})")
        print(f"  Article RT: M={np.mean(art_rts):.1f}s (SD={np.std(art_rts):.1f})")
        print(f"  Difference: {np.mean(art_rts) - np.mean(stmt_rts):.1f}s")
        print(f"  Ratio: {np.mean(art_rts)/np.mean(stmt_rts):.2f}x")
        print(f"\n  Paired t-test: t({len(stmt_rts)-1}) = {t_stat:.3f}, p = {p_value:.4f}")
        print(f"  Cohen's d = {cohens_d:.3f}")
        
        if p_value < 0.05:
            print(f"  [OK] SIGNIFICANT: Articles take longer (deeper processing)")
        else:
            print(f"  - Not significant")
        
        results['h5a'] = {
            'mean_stmt_rt': np.mean(stmt_rts),
            'mean_art_rt': np.mean(art_rts),
            'difference': np.mean(art_rts) - np.mean(stmt_rts),
            'ratio': np.mean(art_rts) / np.mean(stmt_rts),
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }
        
        # H5b: RT differences by article type (if available)
        print("\nH5b: Disconfirmatory RT > Confirmatory RT")
        print("-" * 50)
        print("  ⚠ Requires per-article RT data (not in current summary)")
        print("  -> Implement in detailed behavioral analysis")
        
        results['h5b'] = {'error': 'Per-article RT data not available in summary'}
        
        # H5c: Belief extremity × RT correlation
        print("\nH5c: Belief Extremity × RT Difference")
        print("-" * 50)
        
        if 'mean_belief_extremity' in self.merged_data.columns:
            rt_diff = art_rts - stmt_rts
            
            extremity_rt_r, extremity_rt_p = stats.pearsonr(
                self.merged_data['mean_belief_extremity'].dropna(),
                rt_diff[self.merged_data['mean_belief_extremity'].notna()]
            )
            
            print(f"  Pearson r = {extremity_rt_r:.3f}, p = {extremity_rt_p:.3f}")
            
            if extremity_rt_p < 0.05:
                if extremity_rt_r > 0:
                    print(f"  [OK] SIGNIFICANT: Stronger beliefs -> Longer processing time")
                else:
                    print(f"  [OK] SIGNIFICANT: Stronger beliefs -> Shorter processing (fluency?)")
            else:
                print(f"  - Not significant")
            
            results['h5c'] = {
                'correlation': extremity_rt_r,
                'p_value': extremity_rt_p,
                'significant': extremity_rt_p < 0.05
            }
        else:
            print("  ⚠ Belief extremity data not available")
            results['h5c'] = {'error': 'Data not available'}
        
        return results
    
    def analyze_individual_participants(self):
        """Detailed individual participant profiles"""
        print("\n" + "="*70)
        print("INDIVIDUAL PARTICIPANT PROFILES")
        print("="*70)
        
        individual_results = []
        
        for _, row in self.merged_data.iterrows():
            pid = row['participant_id']
            cb_rate = row['confirmation_bias_rate']
            
            # Z-score relative to baseline
            z_score = (cb_rate - self.CORRECT_BASELINE) / 0.058  # Use Monte Carlo SD
            
            print(f"\nParticipant {pid}")
            print("-" * 50)
            print(f"  CB Rate: {cb_rate:.1%} ({cb_rate/self.CORRECT_BASELINE:.2f}x chance)")
            print(f"  Z-score: {z_score:.2f} ({self._interpret_z_score(z_score)})")
            print(f"  BIS: {row['bis_score']:.2f}, BAS: {row['bas_total']:.2f}")
            
            if 'mean_belief_extremity' in row and pd.notna(row['mean_belief_extremity']):
                print(f"  Belief extremity: {row['mean_belief_extremity']:.2f}")
            
            if 'confidence_numeric' in row and pd.notna(row['confidence_numeric']):
                print(f"  Confidence: {row['confidence_level']}")
            
            if 'cognitive_flexibility' in row and pd.notna(row['cognitive_flexibility']):
                print(f"  Cognitive flexibility: {row['mind_changing_frequency']}")
            
            print(f"  Statement RT: {row['mean_statement_rt']:.1f}s")
            print(f"  Article RT: {row['mean_article_rt']:.1f}s")
            
            individual_results.append({
                'participant_id': pid,
                'cb_rate': cb_rate,
                'z_score': z_score,
                'interpretation': self._interpret_z_score(z_score)
            })
        
        return individual_results
    
    def create_comprehensive_visualizations(self):
        """Create clean separate visualization figures"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        plt.ioff()  # Don't show interactive plots
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        self._plot_cb_rates(fig_dir)
        self._plot_response_times(fig_dir)
        self._plot_bis_bas_profile(fig_dir)
        self._plot_correlation_matrix(fig_dir)
        
        print(f"\n[OK] All figures saved to: {fig_dir}")
    
    def _plot_cb_rates(self, fig_dir):
        """Plot CB rates vs baseline"""
        fig, ax = plt.subplots(figsize=(10, 6))
        participants = [f"P{pid}" for pid in self.merged_data["participant_id"]]
        cb_rates = self.merged_data["confirmation_bias_rate"].values
        
        colors = ["#e74c3c" if r > 0.50 else "#f39c12" if r > 0.35 else "#95a5a6" 
                  for r in cb_rates]
        bars = ax.bar(participants, cb_rates, color=colors, alpha=0.8, edgecolor="black")
        
        ax.axhline(0.208, color="green", linestyle="--", linewidth=2, label="Baseline (20.8%)")
        for bar, rate in zip(bars, cb_rates):
            ax.text(bar.get_x() + bar.get_width()/2, rate, f"{rate:.1%}", 
                   ha="center", va="bottom", fontweight="bold")
        
        ax.set_ylabel("Confirmation Bias Rate", fontsize=12, fontweight="bold")
        ax.set_title("CB Rates vs Monte Carlo Baseline", fontsize=14, fontweight="bold")
        ax.set_ylim(0, max(cb_rates) * 1.2)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "1_cb_rates.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  [OK] CB rates figure")
    
    def _plot_response_times(self, fig_dir):
        """Plot response times"""
        fig, ax = plt.subplots(figsize=(10, 6))
        participants = [f"P{pid}" for pid in self.merged_data["participant_id"]]
        stmt_rt = self.merged_data["mean_statement_rt"].values
        art_rt = self.merged_data["mean_article_rt"].values
        
        x = np.arange(len(participants))
        width = 0.35
        ax.bar(x - width/2, stmt_rt, width, label="Statements", 
               color="#3498db", alpha=0.8, edgecolor="black")
        ax.bar(x + width/2, art_rt, width, label="Articles",
               color="#e67e22", alpha=0.8, edgecolor="black")
        
        ax.set_ylabel("Mean RT (seconds)", fontsize=12, fontweight="bold")
        ax.set_title("Response Times by Phase", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(participants)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "2_response_times.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  [OK] Response times figure")
    
    def _plot_bis_bas_profile(self, fig_dir):
        """Plot BIS/BAS personality profiles"""
        fig, ax = plt.subplots(figsize=(10, 6))
        bis = self.merged_data["bis_score"].values
        bas = self.merged_data["bas_total"].values
        cb_rates = self.merged_data["confirmation_bias_rate"].values
        participants = [f"P{pid}" for pid in self.merged_data["participant_id"]]
        
        scatter = ax.scatter(bis, bas, c=cb_rates, s=300, cmap="RdYlGn_r",
                           edgecolors="black", linewidths=2, alpha=0.8, vmin=0, vmax=1)
        
        for i, txt in enumerate(participants):
            ax.annotate(txt, (bis[i], bas[i]), fontsize=10, fontweight="bold",
                       ha="center", va="center")
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("CB Rate", fontsize=12, fontweight="bold")
        ax.set_xlabel("BIS Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("BAS Score", fontsize=12, fontweight="bold")
        ax.set_title("BIS/BAS Profile by CB Level", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "3_bis_bas_profile.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  [OK] BIS/BAS profile figure")
    
    def _plot_correlation_matrix(self, fig_dir):
        """Plot correlation matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_vars = ["bis_score", "bas_total", "confirmation_bias_rate",
                    "mean_statement_rt", "mean_article_rt"]
        corr_labels = ["BIS", "BAS", "CB Rate", "Stmt RT", "Art RT"]
        corr_data = self.merged_data[corr_vars].corr()
        
        im = ax.imshow(corr_data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(corr_labels)))
        ax.set_yticks(np.arange(len(corr_labels)))
        ax.set_xticklabels(corr_labels, fontsize=11, fontweight="bold")
        ax.set_yticklabels(corr_labels, fontsize=11, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        for i in range(len(corr_labels)):
            for j in range(len(corr_labels)):
                color = "white" if abs(corr_data.iloc[i, j]) > 0.5 else "black"
                ax.text(j, i, f"{corr_data.iloc[i, j]:.2f}", ha="center", va="center",
                       fontsize=10, fontweight="bold", color=color)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation", fontsize=12, fontweight="bold")
        ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(fig_dir / "4_correlation_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  [OK] Correlation matrix figure")
    

    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _interpret_z_score(self, z):
        """Interpret z-score magnitude"""
        abs_z = abs(z)
        if abs_z < 1.0:
            return "within expected range"
        elif abs_z < 1.96:
            return "moderately elevated"
        elif abs_z < 2.58:
            return "significantly elevated (p<0.05)"
        else:
            return "highly significantly elevated (p<0.01)"
    
    def run_complete_analysis(self):
        """Execute complete comprehensive analysis pipeline"""
        print("="*70)
        print("COMPREHENSIVE BCI STATISTICAL ANALYSIS")
        print("="*70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")
        print("="*70)
        
        # 1. Load all data
        print("\n[1/7] Loading questionnaire data...")
        questionnaire_df = self.load_complete_questionnaire_data()
        
        print("\n[2/7] Loading behavioral data...")
        behavioral_df = self.load_behavioral_data()
        
        # 2. Merge datasets
        print("\n[3/7] Merging datasets...")
        questionnaire_df['participant_id'] = questionnaire_df['participant_id'].astype(str)
        behavioral_df['participant_id'] = behavioral_df['participant_id'].astype(str)
        
        self.merged_data = behavioral_df.merge(
            questionnaire_df,
            on='participant_id',
            how='inner'
        )
        
        print(f"  [OK] Successfully merged data for {len(self.merged_data)} participants")
        
        # 3. Test all hypotheses
        print("\n[4/7] Testing hypotheses...")
        hypothesis_results = {}
        
        hypothesis_results['h0'] = self.test_h0_baseline()
        hypothesis_results['h2'] = self.test_h2_bis_bas_effects()
        hypothesis_results['h3'] = self.test_h3_belief_effects()
        hypothesis_results['h4'] = self.test_h4_decision_style_effects()
        hypothesis_results['h5'] = self.test_h5_response_time_patterns()
        
        # 4. Individual analyses
        print("\n[5/7] Analyzing individual participants...")
        individual_results = self.analyze_individual_participants()
        
        # 5. Create visualizations
        print("\n[6/7] Creating comprehensive visualizations...")
        self.create_comprehensive_visualizations()
        
        # 6. Save results
        print("\n[7/7] Saving results...")
        
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_participants': len(self.merged_data),
                'baseline': self.CORRECT_BASELINE,
                'questionnaire_file': self.questionnaire_file,
                'behavioral_file': self.behavioral_file
            },
            'hypothesis_tests': hypothesis_results,
            'individual_analyses': individual_results,
            'summary_statistics': {
                'mean_cb_rate': float(self.merged_data['confirmation_bias_rate'].mean()),
                'std_cb_rate': float(self.merged_data['confirmation_bias_rate'].std()),
                'mean_disconf_rate': float(self.merged_data['disconfirmation_rate'].mean()),
                'mean_statement_rt': float(self.merged_data['mean_statement_rt'].mean()),
                'mean_article_rt': float(self.merged_data['mean_article_rt'].mean()),
            }
        }
        
        # Save JSON results
        json_path = self.output_dir / 'comprehensive_statistical_results.json'
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2, default=float)
        print(f"  [OK] Results saved to {json_path}")
        
        # Save merged data as CSV
        csv_path = self.output_dir / 'merged_participant_data.csv'
        self.merged_data.to_csv(csv_path, index=False)
        print(f"  [OK] Merged data saved to {csv_path}")
        
        # Generate summary report
        self._generate_summary_report(output)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nKey Finding: {hypothesis_results['h0']['reject_h0']}")
        if hypothesis_results['h0']['reject_h0']:
            print("  -> Confirmation bias DETECTED above chance level")
        else:
            print("  -> Insufficient evidence for systematic bias (may be underpowered)")
        
        print(f"\nEffect size: {hypothesis_results['h0']['effect_interpretation']}")
        print(f"Cohen's d = {hypothesis_results['h0']['cohens_d']:.2f}")
        
        return output
    
    def _generate_summary_report(self, results):
        """Generate a text summary report"""
        report_path = self.output_dir / 'ANALYSIS_SUMMARY.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("BCI CONFIRMATION BIAS STUDY - COMPREHENSIVE ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Date: {results['metadata']['timestamp']}\n")
            f.write(f"N = {results['metadata']['n_participants']} participants\n")
            f.write(f"Baseline: {results['metadata']['baseline']:.1%}\n\n")
            
            f.write("HYPOTHESIS TESTING RESULTS:\n")
            f.write("-"*70 + "\n\n")
            
            # H0
            h0 = results['hypothesis_tests']['h0']
            f.write(f"H0: Null Hypothesis (No systematic bias)\n")
            f.write(f"  Decision: {'REJECT' if h0['reject_h0'] else 'FAIL TO REJECT'}\n")
            f.write(f"  Observed: {h0['mean_cb']:.1%} ({h0['ratio']:.2f}x baseline)\n")
            f.write(f"  p-value: {h0['p_one_tailed']:.4f}\n")
            f.write(f"  Effect size: {h0['effect_interpretation']} (d={h0['cohens_d']:.2f})\n\n")
            
            # H2
            h2 = results['hypothesis_tests']['h2']
            f.write(f"H2: BIS/BAS Personality Effects\n")
            f.write(f"  H2a (BIS->CB): r={h2['h2a']['correlation']:.3f}, ")
            f.write(f"p={h2['h2a']['p_value']:.3f} ")
            f.write(f"{'[OK] SIG' if h2['h2a']['significant'] else '- NS'}\n")
            h2b_corr = h2['h2b']['correlation']
            h2b_p = h2['h2b']['p_value']
            if h2b_corr is not None:
                f.write(f"  H2b (BAS->Disconf): r={h2b_corr:.3f}, p={h2b_p:.3f} ")
                f.write(f"{'[OK] SIG' if h2['h2b']['significant'] else '- NS'}\n\n")
            else:
                f.write("  H2b (BAS->Disconf): No variance\n\n")
            f.write(f"p={h2['h2b']['p_value']:.3f} ")
            f.write(f"{'[OK] SIG' if h2['h2b']['significant'] else '- NS'}\n\n")
            
            # H3
            h3 = results['hypothesis_tests']['h3']
            f.write(f"H3: Topic Belief Effects\n")
            if 'correlation' in h3.get('h3a', {}):
                f.write(f"  H3a (Extremity->CB): r={h3['h3a']['correlation']:.3f}, ")
                f.write(f"p={h3['h3a']['p_value']:.3f} ")
                f.write(f"{'[OK] SIG' if h3['h3a']['significant'] else '- NS'}\n")
            else:
                f.write(f"  H3a: Data not available\n")
            
            if 'correlation' in h3.get('h3b', {}):
                f.write(f"  H3b (Familiarity->Bias): r={h3['h3b']['correlation']:.3f}, ")
                f.write(f"p={h3['h3b']['p_value']:.3f} ")
                f.write(f"{'[OK] SIG' if h3['h3b']['significant'] else '- NS'}\n\n")
            else:
                f.write(f"  H3b: Data not available\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS:\n")
            f.write("-"*70 + "\n")
            stats = results['summary_statistics']
            f.write(f"  Mean CB rate: {stats['mean_cb_rate']:.1%} (SD={stats['std_cb_rate']:.1%})\n")
            f.write(f"  Mean disconfirmation rate: {stats['mean_disconf_rate']:.1%}\n")
            f.write(f"  Mean statement RT: {stats['mean_statement_rt']:.1f}s\n")
            f.write(f"  Mean article RT: {stats['mean_article_rt']:.1f}s\n")
            f.write(f"  Article/Statement RT ratio: {stats['mean_article_rt']/stats['mean_statement_rt']:.2f}x\n\n")
            
            f.write("="*70 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("="*70 + "\n")
        
        print(f"  [OK] Summary report saved to {report_path}")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    questionnaire_file = sys.argv[1] if len(sys.argv) > 1 else 'tests_questionnaire_responses.xlsx'
    behavioral_file = sys.argv[2] if len(sys.argv) > 2 else './bci_output/patched/batch_behavioural/batch_summary.csv'
    monte_carlo_baseline = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2081
    
    # Run analysis
    analyzer = ComprehensiveBCIStatisticalAnalysis(
        questionnaire_file=questionnaire_file,
        behavioral_file=behavioral_file,
        monte_carlo_baseline=monte_carlo_baseline,
        output_dir='./bci_output/patched/batch_behavioural/statistical_analysis'
    )
    
    results = analyzer.run_complete_analysis()
    
    print("\n" + "="*70)
    print("All outputs saved to: ./bci_output/patched/batch_behavioural/statistical_analysis/")

    print("="*70)
