#!/usr/bin/env python3
"""
Monte Carlo Baseline for BCI Confirmation Bias Study - CORRECTED METHODOLOGY

VERSION: 4.0 - November 2, 2025 (CORRECTED)

CRITICAL CHANGE:
- Removes attention check filtering from baseline simulation
- CB% = confirmatory / total_articles (not just valid)
- Represents TRUE null hypothesis (random selection without filtering)
- Cap at 10 articles (minimum required to complete experiment)

This fixes the inflated 71% baseline issue.

Author: Jason Stewart
Ethics: ETH23-7909
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List, Dict
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# IMPORTANT: Update these with your ACTUAL rating distributions!
# Run check_rating_distribution.py first, then update these values
STATEMENT_RATING_PROBS = [0.107, 0.129, 0.213, 0.312, 0.239]  # Updated from check_rating_distribution.py for n=6
ARTICLE_RATING_PROBS = [0.015, 0.221, 0.103, 0.308, 0.353]     # Updated from check_rating_distribution.py for n=6

# Experimental design parameters
N_STATEMENTS_TOTAL = 60  # 5 statements × 12 topics
N_ARTICLES_REQUIRED = 10  # Minimum required to complete experiment
N_TOPICS = 12

TOPICS = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 
          'T15', 'T16', 'T17', 'T18', 'T20']

# EXACT ARTICLE-STATEMENT MAPPINGS FROM YOUR EXPERIMENTAL DESIGN
ARTICLE_STATEMENT_MAP = {
    # Topic T01: Climate Change
    'T01A': {'type': 'confirmatory', 'statement': 'T01-S01'},
    'T01B': {'type': 'disconfirmatory', 'statement': 'T01-S01'},
    'T01C': {'type': 'confirmatory', 'statement': 'T01-S02'},
    'T01D': {'type': 'disconfirmatory', 'statement': 'T01-S02'},
    'T01E': {'type': 'neutral', 'statement': 'multiple'},
    
    # Topic T02: Technology
    'T02A': {'type': 'confirmatory', 'statement': 'T02-S01'},
    'T02B': {'type': 'disconfirmatory', 'statement': 'T02-S01'},
    'T02C': {'type': 'confirmatory', 'statement': 'T02-S02'},
    'T02D': {'type': 'disconfirmatory', 'statement': 'T02-S02'},
    'T02E': {'type': 'neutral', 'statement': 'multiple'},
    
    # Topic T03: Economic Policy
    'T03A': {'type': 'confirmatory', 'statement': 'T03-S04'},
    'T03B': {'type': 'disconfirmatory', 'statement': 'T03-S04'},
    'T03C': {'type': 'confirmatory', 'statement': 'T03-S05'},
    'T03D': {'type': 'disconfirmatory', 'statement': 'T03-S05'},
    'T03E': {'type': 'neutral', 'statement': 'multiple'},
    
    # Topic T04: Health
    'T04A': {'type': 'confirmatory', 'statement': 'T04-S01'},
    'T04B': {'type': 'disconfirmatory', 'statement': 'T04-S01'},
    'T04C': {'type': 'confirmatory', 'statement': 'T04-S03'},
    'T04D': {'type': 'disconfirmatory', 'statement': 'T04-S03'},
    'T04E': {'type': 'neutral', 'statement': 'multiple'},
    
    # Topic T05: Education
    'T05A': {'type': 'confirmatory', 'statement': 'T05-S01'},
    'T05B': {'type': 'disconfirmatory', 'statement': 'T05-S01'},
    'T05C': {'type': 'confirmatory', 'statement': 'T05-S02'},
    'T05D': {'type': 'disconfirmatory', 'statement': 'T05-S02'},
    'T05E': {'type': 'neutral', 'statement': 'multiple'},
    
    # Topic T06: AI & Ethics
    'T06A': {'type': 'confirmatory', 'statement': 'T06-S01'},
    'T06B': {'type': 'disconfirmatory', 'statement': 'T06-S01'},
    'T06C': {'type': 'confirmatory', 'statement': 'T06-S02'},
    'T06D': {'type': 'disconfirmatory', 'statement': 'T06-S02'},
    'T06E': {'type': 'neutral', 'statement': 'multiple'},
    
    # Topic T07: Work-Life Balance
    'T07A': {'type': 'confirmatory', 'statement': 'T07-S01'},
    'T07B': {'type': 'disconfirmatory', 'statement': 'T07-S01'},
    'T07C': {'type': 'confirmatory', 'statement': 'T07-S02'},
    'T07D': {'type': 'disconfirmatory', 'statement': 'T07-S02'},
    'T07E': {'type': 'neutral', 'statement': 'multiple'},
    
    # Topic T15: Media & Information
    'T15A': {'type': 'confirmatory', 'statement': 'T15-S01'},
    'T15B': {'type': 'disconfirmatory', 'statement': 'T15-S01'},
    'T15C': {'type': 'confirmatory', 'statement': 'T15-S02'},
    'T15D': {'type': 'disconfirmatory', 'statement': 'T15-S02'},
    'T15E': {'type': 'neutral', 'statement': 'multiple'},
    
    # Topic T16: Gender
    'T16A': {'type': 'confirmatory', 'statement': 'T16-S01'},
    'T16B': {'type': 'disconfirmatory', 'statement': 'T16-S01'},
    'T16C': {'type': 'confirmatory', 'statement': 'T16-S02'},
    'T16D': {'type': 'disconfirmatory', 'statement': 'T16-S02'},
    'T16E': {'type': 'neutral', 'statement': 'multiple'},
    
    # Topic T17: Parenting
    'T17A': {'type': 'confirmatory', 'statement': 'T17-S01'},
    'T17B': {'type': 'disconfirmatory', 'statement': 'T17-S01'},
    'T17C': {'type': 'confirmatory', 'statement': 'T17-S02'},
    'T17D': {'type': 'disconfirmatory', 'statement': 'T17-S02'},
    'T17E': {'type': 'neutral', 'statement': 'multiple'},
    
    # Topic T18: Aging
    'T18A': {'type': 'confirmatory', 'statement': 'T18-S01'},
    'T18B': {'type': 'disconfirmatory', 'statement': 'T18-S01'},
    'T18C': {'type': 'confirmatory', 'statement': 'T18-S02'},
    'T18D': {'type': 'disconfirmatory', 'statement': 'T18-S02'},
    'T18E': {'type': 'neutral', 'statement': 'multiple'},
    
    # Topic T20: Mental Health
    'T20A': {'type': 'confirmatory', 'statement': 'T20-S01'},
    'T20B': {'type': 'disconfirmatory', 'statement': 'T20-S01'},
    'T20C': {'type': 'confirmatory', 'statement': 'T20-S02'},
    'T20D': {'type': 'disconfirmatory', 'statement': 'T20-S02'},
    'T20E': {'type': 'neutral', 'statement': 'multiple'},
}

# ============================================================================
# BIAS CLASSIFICATION
# ============================================================================

def determine_bias_type_exact(stmt_rating: int, article_rating: int, article_code: str) -> str:
    """
    Classify bias type based on statement rating, article rating, and article type.
    Matches the classification logic from bci_analysis_behavioural.py
    """
    article_info = ARTICLE_STATEMENT_MAP.get(article_code, {})
    article_type = article_info.get('type', 'unknown')
    
    if article_type == "neutral":
        return "NEUTRAL"
    
    # High original agreement (agreed with statement)
    if stmt_rating >= 4:
        if article_type == "confirmatory" and article_rating >= 4:
            return "CONFIRMATION_BIAS"
        elif article_type == "disconfirmatory" and article_rating >= 4:
            return "DISCONFIRMATION_SEEKING"
        elif article_type == "confirmatory" and article_rating <= 2:
            return "INCONSISTENT_REJECTION"
        elif article_type == "disconfirmatory" and article_rating <= 2:
            return "CONSISTENT_REJECTION"
    
    # Low original agreement (disagreed with statement)
    elif stmt_rating <= 2:
        if article_type == "disconfirmatory" and article_rating >= 4:
            return "CONFIRMATION_BIAS"
        elif article_type == "confirmatory" and article_rating >= 4:
            return "DISCONFIRMATION_SEEKING"
        elif article_type == "disconfirmatory" and article_rating <= 2:
            return "INCONSISTENT_REJECTION"
        elif article_type == "confirmatory" and article_rating <= 2:
            return "CONSISTENT_REJECTION"
    
    # Neutral original agreement
    else:
        return "NEUTRAL"
    
    return "UNKNOWN"

# ============================================================================
# CORRECTED SIMULATION (v4.0) - NO ATTENTION FILTERING
# ============================================================================

def simulate_participant_no_attention() -> Dict:
    """
    Simulate single participant WITHOUT attention check filtering.
    
    This represents the TRUE null hypothesis: random article selection
    without any filtering based on response quality.
    
    Experimental parameters:
    - 60 statements rated (all 5 per topic × 12 topics)
    - 10 articles selected (minimum required to complete)
    - No attention check filtering
    - CB% = confirmatory_articles / total_articles
    
    Returns:
        Dict with confirmation_rate, disconfirmation_rate, n_total, n_classified
    """
    
    # STEP 1: Generate ratings for ALL 60 statements
    statement_ratings = {}
    for topic in TOPICS:
        for s_num in range(1, 6):  # S01-S05
            stmt_code = f"{topic}-S0{s_num}"
            statement_ratings[stmt_code] = np.random.choice(
                [1, 2, 3, 4, 5], 
                p=STATEMENT_RATING_PROBS
            )
    
    # STEP 2: Select exactly 10 articles (minimum required)
    all_article_codes = list(ARTICLE_STATEMENT_MAP.keys())
    n_articles = N_ARTICLES_REQUIRED  # Fixed at 10
    selected_articles = np.random.choice(all_article_codes, n_articles, replace=False)
    
    # STEP 3: Classify ALL selected articles
    all_classifications = []
    
    for article_code in selected_articles:
        article_info = ARTICLE_STATEMENT_MAP[article_code]
        
        # Get primary linked statement
        primary_statement = article_info.get('statement')
        
        if primary_statement and primary_statement != 'multiple':
            linked_statement = primary_statement
        elif primary_statement == 'multiple':
            # For neutral articles, randomly select a statement from same topic
            topic = article_code[:3]
            linked_statement = f"{topic}-S0{np.random.randint(1, 6)}"
        else:
            continue
        
        # Classify using statement and article ratings
        if linked_statement in statement_ratings:
            stmt_rating = statement_ratings[linked_statement]
            article_rating = np.random.choice([1, 2, 3, 4, 5], p=ARTICLE_RATING_PROBS)
            
            bias_type = determine_bias_type_exact(stmt_rating, article_rating, article_code)
            
            # Keep ALL classifications (not just CB/disconfirm)
            all_classifications.append(bias_type)
    
    # STEP 4: Calculate metrics from TOTAL articles (not just classifiable)
    if n_articles == 0:
        return {
            'confirmation_rate': 0,
            'disconfirmation_rate': 0,
            'n_total': 0,
            'n_classified': 0
        }
    
    cb_count = sum(1 for c in all_classifications if c == "CONFIRMATION_BIAS")
    disconfirm_count = sum(1 for c in all_classifications if c == "DISCONFIRMATION_SEEKING")
    
    # ✅ CORRECTED: Divide by TOTAL articles selected
    confirmation_rate = cb_count / n_articles
    disconfirmation_rate = disconfirm_count / n_articles
    
    return {
        'confirmation_rate': confirmation_rate,
        'disconfirmation_rate': disconfirmation_rate,
        'n_total': n_articles,
        'n_classified': len(all_classifications)
    }


def monte_carlo_no_attention(
    n_simulations: int = 100000, 
    n_participants: int = 6
) -> Tuple[float, float, np.ndarray, Dict]:
    """
    Monte Carlo simulation WITHOUT attention check filtering.
    
    Represents TRUE null hypothesis: random article selection with no
    filtering based on response quality.
    
    Expected baseline: 15-35% (much lower than the inflated 71%)
    
    Args:
        n_simulations: Number of experiments to simulate (default 100,000)
        n_participants: Participants per experiment (default 6)
    
    Returns:
        Tuple of (mean_cb, std_cb, distribution, stats_dict)
    """
    print(f"\n{'='*70}")
    print(f"MONTE CARLO SIMULATION - CORRECTED METHODOLOGY (v4.0)")
    print(f"{'='*70}")
    print(f"Running {n_simulations:,} simulations...")
    print(f"Participants per simulation: {n_participants}")
    print(f"Statements per participant: {N_STATEMENTS_TOTAL} (all rated)")
    print(f"Articles per participant: {N_ARTICLES_REQUIRED} (minimum required)")
    print(f"CB% denominator: TOTAL articles selected (not filtered)")
    print(f"Methodology: TRUE null hypothesis (no attention filtering)")
    print(f"{'='*70}")
    
    experiment_cb_rates = []
    experiment_disconfirm_rates = []
    
    for sim in range(n_simulations):
        if sim % 10000 == 0 and sim > 0:
            print(f"  Progress: {sim:,} / {n_simulations:,} simulations ({sim/n_simulations*100:.0f}%)")
        
        participant_cb_rates = []
        participant_disconfirm_rates = []
        
        for _ in range(n_participants):
            result = simulate_participant_no_attention()
            if result['n_total'] > 0:
                participant_cb_rates.append(result['confirmation_rate'])
                participant_disconfirm_rates.append(result['disconfirmation_rate'])
        
        if participant_cb_rates:
            experiment_cb_rates.append(np.mean(participant_cb_rates))
            experiment_disconfirm_rates.append(np.mean(participant_disconfirm_rates))
    
    cb_distribution = np.array(experiment_cb_rates)
    disconfirm_distribution = np.array(experiment_disconfirm_rates)
    
    mean_cb = np.mean(cb_distribution)
    std_cb = np.std(cb_distribution)
    mean_disconfirm = np.mean(disconfirm_distribution)
    std_disconfirm = np.std(disconfirm_distribution)
    
    # Calculate percentiles
    percentiles = {
        '5th': np.percentile(cb_distribution, 5),
        '25th': np.percentile(cb_distribution, 25),
        '50th': np.percentile(cb_distribution, 50),
        '75th': np.percentile(cb_distribution, 75),
        '95th': np.percentile(cb_distribution, 95)
    }
    
    print(f"\n{'='*70}")
    print(f"SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nConfirmation Bias Baseline:")
    print(f"  Mean: {mean_cb:.4f} ({mean_cb*100:.2f}%)")
    print(f"  SD:   {std_cb:.4f} ({std_cb*100:.2f}%)")
    print(f"  95% CI: [{percentiles['5th']:.3f}, {percentiles['95th']:.3f}]")
    
    print(f"\nDisconfirmation Seeking Baseline:")
    print(f"  Mean: {mean_disconfirm:.4f} ({mean_disconfirm*100:.2f}%)")
    print(f"  SD:   {std_disconfirm:.4f} ({std_disconfirm*100:.2f}%)")
    
    print(f"\n{'='*70}")
    
    # Check if baseline is reasonable
    if mean_cb > 0.50:
        print("⚠️  WARNING: Baseline >50% - check rating distributions!")
        print("   Run check_rating_distribution.py and update RATING_PROBS")
    elif mean_cb < 0.10:
        print("⚠️  WARNING: Baseline <10% - seems too low, check implementation")
    else:
        print("✅ Baseline in expected range (10-50%)")
    
    simulation_stats = {
        'cb_mean': float(mean_cb),
        'cb_std': float(std_cb),
        'cb_percentiles': {k: float(v) for k, v in percentiles.items()},
        'disconfirm_mean': float(mean_disconfirm),
        'disconfirm_std': float(std_disconfirm),
        'n_simulations': n_simulations,
        'n_participants': n_participants,
        'methodology': 'no_attention_filtering_v4.0'
    }
    
    return mean_cb, std_cb, cb_distribution, simulation_stats


def create_visualization(distribution: np.ndarray, mean: float, std: float, 
                         observed: float = None, output_file: str = 'monte_carlo_corrected.png'):
    """Create visualization of Monte Carlo results"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot distribution
    ax.hist(distribution, bins=50, density=True, alpha=0.7, 
            color='steelblue', edgecolor='black', label='Null Distribution')
    
    # Add normal curve
    x = np.linspace(distribution.min(), distribution.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2, 
            label=f'Normal(μ={mean:.3f}, σ={std:.3f})')
    
    # Add mean line
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, 
               label=f'Baseline Mean: {mean:.1%}')
    
    # Add observed value if provided
    if observed is not None:
        ax.axvline(observed, color='green', linestyle='-', linewidth=2,
                  label=f'Observed: {observed:.1%}')
        
        # Calculate z-score and p-value
        z_score = (observed - mean) / std
        p_value = 1 - stats.norm.cdf(z_score)
        
        ax.text(0.02, 0.98, 
                f'Z-score: {z_score:.2f}\np-value: {p_value:.4f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    ax.set_xlabel('Confirmation Bias Rate', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Monte Carlo Baseline Distribution (Corrected - No Attention Filtering)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualization saved: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BCI CONFIRMATION BIAS STUDY - CORRECTED MONTE CARLO BASELINE")
    print("="*70)
    print("\nVersion 4.0 - METHODOLOGICAL CORRECTION")
    print("\nKey changes from v3.0:")
    print("  - Removed attention check filtering")
    print("  - CB% = confirmatory / total_articles (all 10 articles)")
    print("  - Represents TRUE null hypothesis (random selection)")
    print("  - Expected baseline: 15-35% (down from 71%)")
    
    # Check if rating distributions have been updated
    if STATEMENT_RATING_PROBS == [0.20, 0.20, 0.20, 0.20, 0.20]:
        print("\n⚠️  WARNING: Using default rating distributions!")
        print("   Run check_rating_distribution.py and update lines 26-27")
        print("   Press Enter to continue anyway, or Ctrl+C to abort...")
        input()
    
    # Run corrected simulation
    mean_cb, std_cb, distribution, simulation_stats = monte_carlo_no_attention(
        n_simulations=100000,
        n_participants=6
    )
    
    # Save results
    results = {
        'version': '4.0_corrected',
        'date': '2025-11-02',
        'methodology': 'CB% = confirmatory / total_articles (no attention filtering)',
        'rationale': 'Separates selection behavior (CB metric) from response quality (attention checks)',
        'experimental_parameters': {
            'n_statements': N_STATEMENTS_TOTAL,
            'n_articles': N_ARTICLES_REQUIRED,
            'n_topics': N_TOPICS,
            'article_selection': 'minimum required to complete experiment',
            'statement_rating_probs': STATEMENT_RATING_PROBS,
            'article_rating_probs': ARTICLE_RATING_PROBS
        },
        'baseline': simulation_stats,
        'interpretation': 'Expected CB rate under random selection without attention filtering'
    }
    
    output_file = 'monte_carlo_corrected_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved: {output_file}")
    
    # Create visualization (optionally include observed value)
    # Update this line with your actual observed CB rate after running behavioral analysis
    OBSERVED_CB_RATE = 0.5256  # Set to your observed rate, e.g., 0.396
    
    create_visualization(distribution, mean_cb, std_cb, OBSERVED_CB_RATE)
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"CORRECTED BASELINE FOR STATISTICAL ANALYSIS")
    print(f"{'='*70}")
    print(f"\nUse this baseline value: {mean_cb:.3f} ({mean_cb*100:.1f}%)")
    print(f"Standard deviation: {std_cb:.3f}")
    print(f"\nThis represents the TRUE null hypothesis:")
    print(f"  - Random article selection")
    print(f"  - No attention check filtering")
    print(f"  - Minimum 10 articles per participant")
    print(f"  - All 60 statements rated")
    
    print(f"\n{'='*70}")
    print(f"NEXT STEPS:")
    print(f"{'='*70}")
    print(f"1. Update bci_analysis_behavioural.py (remove attention filtering)")
    print(f"2. Re-run behavioral analysis on all participants")
    print(f"3. Update OBSERVED_CB_RATE in this script (line 440)")
    print(f"4. Re-run this script to generate comparison visualization")
    print(f"5. Run statistical analysis with corrected baseline")
    print(f"{'='*70}\n")