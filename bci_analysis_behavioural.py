#!/usr/bin/env python3
"""
Streamlined BCI Analysis Pipeline for Confirmation Bias Study
Standalone script with embedded XDF reconstruction and proper RT calculation

VERSION 4.0 - CORRECTED METHODOLOGY (November 2, 2025)
Changes from v3.0:
- Removed attention check filtering from CB% calculation
- CB% = confirmatory / total_articles (measures selection behavior)
- Attention checks tracked separately for EEG epoch rejection, participant exclusion, quality reporting
- Added new metrics: n_total_articles, classifiable_rate
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import pyxdf
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Import article mappings (only external dependency)
try:
    from article_statement_mappings import (
        ARTICLE_STATEMENT_MAP, 
        get_article_info,
        get_article_relationships,
        get_primary_statement_advanced
    )
except ImportError:
    print("Warning: article_statement_mappings.py not found, using embedded mappings")
    # Embedded minimal mappings as fallback
    ARTICLE_STATEMENT_MAP = {}
    def get_article_info(code): return {'statement': None, 'type': 'unknown', 'weight': 0}
    def get_article_relationships(code): return {}
    def get_primary_statement_advanced(code): return None, 0, 'unknown'

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class StatementResponse:
    """Statement response with proper timing"""
    topic_code: str
    statement_code: str
    agreement: int
    attention_check: str
    presented_time: float  # When statement was shown
    response_time: float   # When response was made
    reaction_time: float   # Calculated difference

@dataclass
class ArticleResponse:
    """Article response with proper timing"""
    article_code: str
    topic_code: str
    agreement: int
    attention_check: str
    start_time: float      # When article reading started
    rating_time: float     # When rating was made
    reading_time: float    # Calculated difference

# ============================================================================
# XDF RECONSTRUCTION (Embedded)
# ============================================================================

class XDFReconstructor:
    """Reconstruct behavioral data from XDF markers with proper RT calculation"""
    
    def __init__(self, xdf_file: str):
        self.streams, self.header = pyxdf.load_xdf(xdf_file)
        self.marker_stream = None
        
        # Find marker stream
        for stream in self.streams:
            if 'Marker' in stream['info']['name'][0]:
                self.marker_stream = stream
                break
    
    def reconstruct_responses(self) -> Tuple[List[StatementResponse], List[ArticleResponse]]:
        """Reconstruct all responses with proper reaction times"""
        if not self.marker_stream:
            return [], []
        
        statements = []
        articles = []
        processed_articles = set()  # Track processed articles to prevent duplicates
        
        # Track presentation times for RT calculation
        statement_presentations = {}  # question_index -> timestamp
        article_starts = {}           # article_code -> timestamp
        
        markers = list(zip(self.marker_stream['time_stamps'], 
                          self.marker_stream['time_series']))
        
        current_statement = None
        current_article = None
        
        for i, (timestamp, marker_data) in enumerate(markers):
            marker_str = marker_data[0] if isinstance(marker_data, (list, np.ndarray)) else marker_data
            
            # === STATEMENT PROCESSING ===
            
            # Statement presented - record start time
            if 'STATEMENT_PRESENTED' in marker_str:
                match = re.match(r'STATEMENT_PRESENTED_Q(\d+)_([A-Z0-9]+)_([A-Z0-9]+)', marker_str)
                if match:
                    q_idx = int(match.group(1))
                    topic = match.group(2)
                    stmt_code = match.group(3)
                    
                    # Store presentation time
                    statement_presentations[q_idx] = timestamp
                    
                    # Create statement object
                    current_statement = {
                        'topic_code': topic,
                        'statement_code': f"{topic}-{stmt_code}",
                        'presented_time': timestamp,
                        'q_index': q_idx
                    }
            
            # Statement response - calculate RT
            elif 'STATEMENT_RESPONSE' in marker_str and 'AGREEMENT' in marker_str:
                match = re.match(r'STATEMENT_RESPONSE_Q(\d+)_([A-Z0-9]+)_([A-Z0-9]+)_AGREEMENT_(\d+)', marker_str)
                if match and current_statement:
                    q_idx = int(match.group(1))
                    agreement = int(match.group(4))
                    
                    # Calculate reaction time as difference
                    presented_time = statement_presentations.get(q_idx, timestamp)
                    reaction_time = timestamp - presented_time
                    
                    current_statement['agreement'] = agreement
                    current_statement['response_time'] = timestamp
                    current_statement['reaction_time'] = reaction_time
            
            # Attention check for statement
            elif 'ATTENTION_CHECK_RESPONSE' in marker_str and current_statement:
                # Both YES and NO are correct responses (participant engaged)
                if '_YES' in marker_str or '_NO' in marker_str:
                    attention = 'PASS'
                elif 'ATTENTION_FAIL' in marker_str:
                    attention = 'FAIL'
                else:
                    attention = 'UNKNOWN'
                
                # Complete statement response
                if 'agreement' in current_statement:
                    statements.append(StatementResponse(
                        topic_code=current_statement['topic_code'],
                        statement_code=current_statement['statement_code'],
                        agreement=current_statement['agreement'],
                        attention_check=attention,
                        presented_time=current_statement['presented_time'],
                        response_time=current_statement['response_time'],
                        reaction_time=current_statement['reaction_time']
                    ))
                current_statement = None
            
            # === ARTICLE PROCESSING ===
            
            # Article reading started - record start time
            elif 'ARTICLE_READ_START' in marker_str:
                match = re.search(r'ARTICLE_READ_START_([A-Z0-9]+)', marker_str)
                if match:
                    article_code = match.group(1)
                    article_starts[article_code] = timestamp
                    # Don't create current_article here to avoid confusion
            
            # Article rating - calculate reading time
            elif 'ARTICLE_RATING' in marker_str:
                match = re.match(r'ARTICLE_RATING_([A-Z0-9]+)_R(\d+)', marker_str)
                if match:
                    article_code = match.group(1)
                    agreement = int(match.group(2))
                    
                    # Skip if already processed
                    if article_code in processed_articles:
                        continue
                    
                    # Calculate reading time as difference
                    start_time = article_starts.get(article_code, timestamp - 5.0)  # Default 5s if no start marker
                    reading_time = timestamp - start_time
                    
                    # Create article response
                    topic_code = article_code[:3] if len(article_code) >= 3 else ''
                    
                    # Save the current article, will add to list when we get attention check or immediately
                    current_article = {
                        'article_code': article_code,
                        'topic_code': topic_code,
                        'agreement': agreement,
                        'start_time': start_time,
                        'rating_time': timestamp,
                        'reading_time': reading_time,
                        'attention_check': 'PASS'  # Default to PASS, will be updated if check follows
                    }
                    
                    # Check if attention check response follows within next 10 markers
                    # (there may be intermediate markers like ACTUAL_BIAS, ATTENTION_CHECK_START)
                    next_is_attention = False
                    lookahead_range = min(10, len(markers) - i - 1)
                    for j in range(1, lookahead_range + 1):
                        next_marker = markers[i + j][1]
                        next_marker_str = next_marker[0] if isinstance(next_marker, (list, np.ndarray)) else next_marker
                        if 'ATTENTION_CHECK_RESPONSE' in str(next_marker_str):
                            next_is_attention = True
                            break
                        # Stop looking if we hit another article or statement
                        if 'ARTICLE_RATING' in str(next_marker_str) or 'STATEMENT_PRESENTED' in str(next_marker_str):
                            break
                    
                    # If no attention check response follows, save immediately
                    if not next_is_attention:
                        articles.append(ArticleResponse(
                            article_code=current_article['article_code'],
                            topic_code=current_article['topic_code'],
                            agreement=current_article['agreement'],
                            attention_check=current_article['attention_check'],
                            start_time=current_article['start_time'],
                            rating_time=current_article['rating_time'],
                            reading_time=current_article['reading_time']
                        ))
                        processed_articles.add(article_code)  # Mark as processed
                        current_article = None  # Clear to prevent duplicate saving
            
            # Article attention check - Support both Phase 1 and Phase 2 formats
            # CRITICAL: Only process RESPONSE markers, not START markers
            elif 'ATTENTION_CHECK_RESPONSE' in marker_str and current_article and isinstance(current_article, dict):
                # Get article code from current article
                article_code = current_article.get('article_code', '')
                
                # Skip if already processed
                if article_code in processed_articles:
                    current_article = None
                    continue
                
                # Determine attention check result
                if 'PHASE2' in marker_str:
                    # Phase 2 format: ATTENTION_CHECK_RESPONSE_PHASE2_CORRECT/INCORRECT
                    # MUST check INCORRECT before CORRECT (substring matching)
                    if 'INCORRECT' in marker_str:
                        attention = 'FAIL'
                    elif 'CORRECT' in marker_str:
                        attention = 'PASS'
                    else:
                        attention = 'UNKNOWN'
                else:
                    # Phase 1 format: ATTENTION_CHECK_..._YES/NO (legacy)
                    if '_YES' in marker_str or '_NO' in marker_str:
                        attention = 'PASS'  # Both YES and NO count as correct response
                    else:
                        attention = 'FAIL'
                
                # Update the current article's attention check
                current_article['attention_check'] = attention
                
                # Now save the complete article
                articles.append(ArticleResponse(
                    article_code=current_article['article_code'],
                    topic_code=current_article['topic_code'],
                    agreement=current_article['agreement'],
                    attention_check=attention,
                    start_time=current_article['start_time'],
                    rating_time=current_article['rating_time'],
                    reading_time=current_article['reading_time']
                ))
                processed_articles.add(article_code)  # Mark as processed
                current_article = None
        
        return statements, articles

# ============================================================================
# JSON DATA LOADER
# ============================================================================

def load_json_data(json_file: str) -> Tuple[List[StatementResponse], List[ArticleResponse]]:
    """Load behavioral data from JSON export with proper timing extraction"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    statements = []
    articles = []
    
    # Parse statements from 'responses' field (actual JSON structure)
    if 'responses' in data:
        for item in data['responses']:
            # Skip entries with NR (No Response)
            if item.get('selectedOption', '') == 'NR':
                continue
                
            topic_code = item.get('topicCode', '')
            statement_code = item.get('statementCode', '')
            
            # Construct full statement code (e.g., "T01-S01")
            if topic_code and statement_code:
                full_statement_code = f"{topic_code}-{statement_code}"
            else:
                full_statement_code = ''
            
            # Handle potential NR or invalid values
            try:
                agreement_value = int(item.get('selectedOption', 3))
            except (ValueError, TypeError):
                # Skip this entry if selectedOption can't be converted to int
                continue
            
            # Handle reaction time conversion
            try:
                reaction_time = float(item.get('agreementReactionTime', 0))
            except (ValueError, TypeError):
                reaction_time = 0.0
            
            # Convert attention check response to PASS/FAIL format
            raw_attention = item.get('attentionCheckResponse', 'YES')
            if raw_attention in ['YES', 'NO']:
                attention_check = 'PASS'
            elif raw_attention == 'ATTENTION_FAIL':
                attention_check = 'FAIL'
            else:
                attention_check = 'PASS'  # Default
            
            statements.append(StatementResponse(
                topic_code=topic_code,
                statement_code=full_statement_code,
                agreement=agreement_value,
                attention_check=attention_check,
                presented_time=0,  # Not available in JSON
                response_time=reaction_time,
                reaction_time=reaction_time
            ))
    
    # Extract article reading times from eventMarkers if available
    article_timings = {}
    if 'eventMarkers' in data:
        for marker in data['eventMarkers']:
            label = marker.get('label', '')
            timestamp = marker.get('localTimestamp', 0) or marker.get('globalTimestamp', 0)
            
            # Article read button clicked (start time)
            if 'READ_ARTICLE_BUTTON_CLICKED' in label:
                # Extract article name after the colon
                parts = label.split(': ')
                if len(parts) > 1:
                    article_name = parts[1].replace(' (local:', '').split(')')[0].strip()
                    if article_name not in article_timings:
                        article_timings[article_name] = {}
                    article_timings[article_name]['start'] = timestamp
            
            # Article agreement selected (end time)
            elif 'AgreementSelected' in label and 'Headline:' in label:
                # Extract headline
                if 'Headline: ' in label:
                    headline_part = label.split('Headline: ')[1]
                    article_name = headline_part.split(' (local:')[0].strip()
                    if article_name not in article_timings:
                        article_timings[article_name] = {}
                    article_timings[article_name]['end'] = timestamp
    
    # Parse articles from 'selectedArticles' field with timing data
    if 'selectedArticles' in data:
        for item in data['selectedArticles']:
            article_code = item.get('articleCode', '')
            headline = item.get('headline', '')
            
            # Skip entries with invalid selectedOption
            try:
                agreement_value = int(item.get('selectedOption', 3))
            except (ValueError, TypeError):
                # Skip this entry if selectedOption can't be converted to int
                continue
            
            # Calculate reading time from markers if available
            reading_time = 10.0  # Default
            if headline in article_timings:
                timing = article_timings[headline]
                if 'start' in timing and 'end' in timing:
                    reading_time = timing['end'] - timing['start']
                    # Sanity check: reading time should be reasonable
                    if reading_time < 1 or reading_time > 120:
                        reading_time = 10.0  # Fallback to default if unreasonable
            
            # Convert attention check response to PASS/FAIL format
            raw_attention = item.get('attentionCheckResponse', 'YES')
            if raw_attention in ['YES', 'NO']:
                attention_check = 'PASS'
            elif raw_attention == 'ATTENTION_FAIL' or raw_attention == 'INCORRECT':
                attention_check = 'FAIL'
            elif raw_attention == 'CORRECT':
                attention_check = 'PASS'
            else:
                attention_check = 'PASS'  # Default
            
            articles.append(ArticleResponse(
                article_code=article_code,
                topic_code=article_code[:3] if len(article_code) >= 3 else '',
                agreement=agreement_value,
                attention_check=attention_check,
                start_time=0,  # Could extract from markers if needed
                rating_time=reading_time,
                reading_time=reading_time
            ))
    
    return statements, articles

# ============================================================================
# BIAS CALCULATION WITH WEIGHTS
# ============================================================================

def calculate_weighted_alignment(orig_agree: int, article_agree: int, relationship: str) -> float:
    """
    Calculate alignment score based on agreement pattern and relationship type.
    Returns score from -1.0 (counter to relationship) to +1.0 (aligned with relationship)
    
    This captures how well the article response aligns with the relationship type.
    """
    strong_agree = (orig_agree >= 4)
    strong_disagree = (orig_agree <= 2)
    neutral_original = (orig_agree == 3)
    
    article_strong_agree = (article_agree >= 4)
    article_strong_disagree = (article_agree <= 2)
    
    # DIRECT CONFIRMATORY - expects agreement to strengthen or maintain
    if relationship == 'direct-confirm':
        if strong_agree and article_strong_agree:
            return 1.0  # Strong confirmation bias
        elif strong_disagree and article_strong_disagree:
            return 1.0  # Maintained disagreement (confirmatory)
        elif neutral_original and article_strong_agree:
            return 0.5  # Moved to agreement
        elif strong_agree and not article_strong_agree:
            return -0.5  # Weakened agreement (disconfirmatory)
        else:
            return 0.0
    
    # DIRECT DISCONFIRMATORY - expects challenge to be accepted
    elif relationship == 'direct-disconfirm':
        if strong_agree and not article_strong_agree:
            return 1.0  # Disconfirmation seeking (accepted challenge)
        elif strong_disagree and not article_strong_disagree:
            return 1.0  # Disagreement weakened (sought disconfirmation)
        elif strong_agree and article_strong_agree:
            return -1.0  # Maintained despite challenge (confirmation bias)
        elif strong_disagree and article_strong_disagree:
            return -1.0  # Maintained despite challenge
        else:
            return 0.0
    
    # INDIRECT CONFIRMATORY - similar to direct but softer
    elif relationship == 'indirect-confirm':
        if (strong_agree and article_strong_agree) or (strong_disagree and article_strong_disagree):
            return 0.75
        elif neutral_original:
            return 0.3
        else:
            return 0.0
    
    # INDIRECT DISCONFIRMATORY
    elif relationship == 'indirect-disconfirm':
        if strong_agree and not article_strong_agree:
            return 0.75
        elif strong_disagree and not article_strong_disagree:
            return 0.75
        elif (strong_agree and article_strong_agree) or (strong_disagree and article_strong_disagree):
            return -0.75
        else:
            return 0.0
    
    # NEUTRAL - no strong bias expected
    elif relationship == 'neutral':
        shift = abs(article_agree - orig_agree)
        if shift >= 3:
            return -0.3  # Penalize extreme shifts in neutral
        else:
            return 0.0
    
    return 0.0


def determine_bias_type_weighted(orig_agree: int, article_agree: int, 
                                 relationship: str, weight: float) -> str:
    """
    Classify bias type using WEIGHTED relationship information.
    
    Uses relationship type and weight to determine bias classification.
    """
    # Get alignment score
    alignment = calculate_weighted_alignment(orig_agree, article_agree, relationship)
    weighted_score = alignment * weight
    
    # Classify based on weighted score and relationship
    if 'neutral' in relationship:
        return 'NEUTRAL'
    
    # Strong confirmation bias (aligned with confirmatory relationship)
    if weighted_score > 0.5 and 'confirm' in relationship:
        return 'CONFIRMATION_BIAS'
    
    # Moderate confirmation bias
    elif weighted_score > 0.2 and 'confirm' in relationship:
        return 'CONFIRMATION_BIAS'
    
    # Strong disconfirmation seeking (accepted disconfirmatory article)
    elif weighted_score > 0.5 and 'disconfirm' in relationship:
        return 'DISCONFIRMATION_SEEKING'
    
    # Moderate disconfirmation seeking
    elif weighted_score > 0.2 and 'disconfirm' in relationship:
        return 'DISCONFIRMATION_SEEKING'
    
    # Counter-bias (rejected confirmatory or avoided disconfirmatory)
    elif weighted_score < -0.5:
        return 'COUNTER_BIAS'
    
    # Mixed/unclear pattern
    else:
        return 'MIXED'


def calculate_bias_metrics(statements: List[StatementResponse], 
                          articles: List[ArticleResponse]) -> Dict:
    """
    Calculate bias metrics using WEIGHTED article-statement relationships.
    UPDATED: Now uses relationship weights (1.0, 0.5, 0.3) and types properly.
    """
    
    # Create statement lookup
    stmt_dict = {}
    for stmt in statements:
        # Skip statements with failed attention checks (weight = 0)
        if stmt.attention_check == 'PASS':
            stmt_dict[stmt.statement_code] = stmt
    
    # Analyse each article with WEIGHTED scoring
    bias_analyses = []
    
    for article in articles:
        # CORRECTED v4.0: Don't skip based on attention checks for CB metric
        # CB% measures SELECTION behavior, not response quality
        # Attention checks used separately for: (1) EEG epoch rejection, 
        # (2) participant exclusion, (3) quality reporting
        
        # Get article info (backward compatible)
        article_info = get_article_info(article.article_code)
        article_type = article_info['type']
        
        # Get WEIGHTED relationship info
        primary_stmt, primary_weight, primary_relationship = get_primary_statement_advanced(
            article.article_code
        )
        
        # Skip if no linkage
        if not primary_stmt:
            continue
        
        # Handle neutral articles with multiple statements
        if primary_stmt == 'multiple' and article_type == 'neutral':
            # Use average of topic statements
            topic_stmts = [s for s in statements 
                         if s.topic_code == article.topic_code 
                         and s.attention_check == 'PASS']
            if topic_stmts:
                avg_agreement = np.mean([s.agreement for s in topic_stmts])
                # For neutral, just use simple classification
                bias_type = determine_bias_type_weighted(
                    round(avg_agreement), 
                    article.agreement, 
                    primary_relationship,
                    primary_weight
                )
                bias_analyses.append({
                    'article_code': article.article_code,
                    'article_type': article_type,
                    'bias_type': bias_type,
                    'original_agreement': round(avg_agreement),
                    'article_agreement': article.agreement,
                    'primary_relationship': primary_relationship,
                    'relationship_weight': primary_weight,
                    'weighted_bias_score': 0.0  # Neutral
                })
            continue
        
        # Find linked statement
        if primary_stmt not in stmt_dict:
            continue
        
        original_stmt = stmt_dict[primary_stmt]
        
        # Calculate WEIGHTED bias score
        weighted_score = calculate_weighted_alignment(
            original_stmt.agreement,
            article.agreement,
            primary_relationship
        )
        
        # Determine bias type using WEIGHTED classification
        bias_type = determine_bias_type_weighted(
            original_stmt.agreement,
            article.agreement,
            primary_relationship,
            primary_weight
        )
        
        bias_analyses.append({
            'article_code': article.article_code,
            'article_type': article_type,
            'bias_type': bias_type,
            'original_agreement': original_stmt.agreement,
            'article_agreement': article.agreement,
            'linked_statement': primary_stmt,
            'primary_relationship': primary_relationship,
            'relationship_weight': primary_weight,
            'weighted_bias_score': weighted_score * primary_weight  # Apply weight
        })
    
    # CORRECTED v4.0: Calculate metrics from TOTAL articles (not just classifiable)
    total_articles = len(articles)  # ALL articles selected by participant
    
    if bias_analyses:
        bias_types = [b['bias_type'] for b in bias_analyses]
        
        # Count different bias types
        confirmation_count = sum(1 for bt in bias_types if 'CONFIRMATION' in bt)
        disconfirmation_count = sum(1 for bt in bias_types if 'DISCONFIRMATION' in bt)
        
        # CORRECTED: Divide by TOTAL articles (selection behavior)
        confirmation_rate = confirmation_count / total_articles if total_articles > 0 else 0
        disconfirmation_rate = disconfirmation_count / total_articles if total_articles > 0 else 0
        
        # Track what proportion were classifiable (for quality reporting)
        classifiable_rate = len(bias_analyses) / total_articles if total_articles > 0 else 0
        
        # Calculate mean weighted bias score
        weighted_scores = [b['weighted_bias_score'] for b in bias_analyses]
        mean_bias_score = np.mean(weighted_scores) if weighted_scores else 0
        
        # Count valid attention checks (still tracked for quality/EEG)
        total_checks = len(statements) + len(articles)
        passed_checks = (sum(1 for s in statements if s.attention_check == 'PASS') +
                        sum(1 for a in articles if a.attention_check == 'PASS'))
        attention_rate = passed_checks / total_checks if total_checks > 0 else 0
    else:
        confirmation_rate = 0
        disconfirmation_rate = 0
        classifiable_rate = 0
        mean_bias_score = 0
        attention_rate = 0
    
    return {
        # PRIMARY METRICS (corrected methodology v4.0)
        'confirmation_bias_rate': confirmation_rate,          # Now from total articles
        'disconfirmation_seeking_rate': disconfirmation_rate, # Now from total articles
        'weighted_mean_bias_score': mean_bias_score,
        
        # QUALITY METRICS (attention checks for EEG/exclusion/reporting)
        'attention_pass_rate': attention_rate,
        'classifiable_rate': classifiable_rate,  # NEW: proportion with valid linkages
        
        # COUNTS (transparency and secondary analyses)
        'n_total_articles': total_articles,       # NEW: denominator for CB%
        'n_classifiable': len(bias_analyses),     # NEW: how many were classified
        'n_valid_pairs': len(bias_analyses),      # Keep for compatibility
        
        # DETAILED
        'detailed_analyses': bias_analyses
    }

def determine_bias_type(orig_agree: int, article_agree: int, article_type: str) -> str:
    """
    DEPRECATED: Old simplified bias classification (kept for backward compatibility).
    Use determine_bias_type_weighted() instead which properly uses relationship weights.
    """
    if article_type == "neutral":
        return "NEUTRAL"
    
    if orig_agree >= 4:  # Originally agreed
        if article_type == "confirmatory" and article_agree >= 4:
            return "CONFIRMATION_BIAS"
        elif article_type == "disconfirmatory" and article_agree >= 4:
            return "DISCONFIRMATION_SEEKING"
    elif orig_agree <= 2:  # Originally disagreed
        if article_type == "disconfirmatory" and article_agree >= 4:
            return "CONFIRMATION_BIAS"
        elif article_type == "confirmatory" and article_agree >= 4:
            return "DISCONFIRMATION_SEEKING"
    
    return "MIXED"

# ============================================================================
# HELPER PLOTTING FUNCTIONS
# ============================================================================

def plot_response_distribution_by_phase(ax, stmt_df, art_df):
    """Plot response distribution comparison between phases"""
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    
    # Plot histograms
    if not stmt_df.empty:
        ax.hist(stmt_df['agreement'], bins=bins, alpha=0.6, label='Statements', color='blue', edgecolor='black')
    if not art_df.empty:
        ax.hist(art_df['agreement'], bins=bins, alpha=0.6, label='Articles', color='red', edgecolor='black')
    
    ax.set_xlabel('Agreement Level')
    ax.set_ylabel('Frequency')
    ax.set_title('Response Distribution by Phase')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_rt_by_agreement_comparison(ax, stmt_df, art_df):
    """Plot RT by agreement level for both phases"""
    agreement_levels = [1, 2, 3, 4, 5]
    
    # Calculate mean RTs for each level
    stmt_means = []
    stmt_stds = []
    art_means = []
    art_stds = []
    
    for level in agreement_levels:
        # Statements
        level_data = stmt_df[stmt_df['agreement'] == level]['rt']
        stmt_means.append(level_data.mean() if len(level_data) > 0 else 0)
        stmt_stds.append(level_data.std() if len(level_data) > 0 else 0)
        
        # Articles
        level_data = art_df[art_df['agreement'] == level]['rt']
        art_means.append(level_data.mean() if len(level_data) > 0 else 0)
        art_stds.append(level_data.std() if len(level_data) > 0 else 0)
    
    # Plot with error bars
    x = np.array(agreement_levels)
    width = 0.35
    
    ax.bar(x - width/2, stmt_means, width, yerr=stmt_stds, label='Statements', 
           color='blue', alpha=0.7, capsize=5)
    ax.bar(x + width/2, art_means, width, yerr=art_stds, label='Articles', 
           color='red', alpha=0.7, capsize=5)
    
    ax.set_xlabel('Agreement Level')
    ax.set_ylabel('Reaction Time (s)')
    ax.set_title('RT by Agreement Level Comparison')
    ax.set_xticks(agreement_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_bias_distribution(ax, bias_metrics):
    """Plot bias type distribution with abbreviated x-axis labels"""
    if not bias_metrics['detailed_analyses']:
        ax.text(0.5, 0.5, 'No valid data for bias analysis', ha='center', va='center')
        ax.set_title('Bias Type Distribution')
        return
    
    # Count bias types
    bias_types = [b['bias_type'] for b in bias_metrics['detailed_analyses']]
    type_counts = pd.Series(bias_types).value_counts()
    
    # Define colors
    color_map = {
        'CONFIRMATION_BIAS': '#ef4444',
        'DISCONFIRMATION_SEEKING': '#3b82f6', 
        'NEUTRAL': '#eab308',
        'MIXED': '#6b7280'
    }
    
    # Create abbreviated labels for x-axis
    label_map = {
        'CONFIRMATION_BIAS': 'Confirm.\nBias',
        'DISCONFIRMATION_SEEKING': 'Disconf.\nSeeking',
        'NEUTRAL': 'Neutral',
        'MIXED': 'Mixed'
    }
    
    # Get colors and labels for the actual data
    bar_colors = [color_map.get(t, '#6b7280') for t in type_counts.index]
    bar_labels = [label_map.get(t, t) for t in type_counts.index]
    
    # Plot bars
    bars = ax.bar(range(len(type_counts)), type_counts.values, color=bar_colors)
    
    # Set x-axis with abbreviated labels
    ax.set_xticks(range(len(type_counts)))
    ax.set_xticklabels(bar_labels, rotation=0, ha='center', fontsize=8)
    
    ax.set_ylabel('Count')
    ax.set_title('Bias Type Distribution')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, type_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{int(value)}', ha='center', va='bottom')

def plot_rt_by_topic(ax, stmt_df, art_df):
    """Plot average RT comparison by topic"""
    # Get unique topics that appear in both dataframes
    all_topics = sorted(set(stmt_df['topic'].unique()) | set(art_df['topic'].unique()))
    
    stmt_means = []
    art_means = []
    
    for topic in all_topics:
        # Statement RT for topic
        topic_stmts = stmt_df[stmt_df['topic'] == topic]['rt']
        stmt_means.append(topic_stmts.mean() if len(topic_stmts) > 0 else 0)
        
        # Article RT for topic
        topic_arts = art_df[art_df['topic'] == topic]['rt']
        art_means.append(topic_arts.mean() if len(topic_arts) > 0 else 0)
    
    # Create grouped bar chart
    x = np.arange(len(all_topics))
    width = 0.35
    
    ax.bar(x - width/2, stmt_means, width, label='Statements', color='blue', alpha=0.7)
    ax.bar(x + width/2, art_means, width, label='Articles', color='red', alpha=0.7)
    
    ax.set_xlabel('Topic')
    ax.set_ylabel('Average Reaction Time (s)')
    ax.set_title('Average Reaction Time by Topic')
    ax.set_xticks(x)
    ax.set_xticklabels(all_topics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_individual_rts_for_topic(ax, topic, stmt_df, art_df):
    """Plot individual statement and article RTs for a specific topic"""
    topic_stmts = stmt_df[stmt_df['topic'] == topic]
    topic_arts = art_df[art_df['topic'] == topic]
    
    # Prepare data for plotting
    stmt_items = []
    art_items = []
    
    # Get individual statements (show full codes)
    for _, row in topic_stmts.iterrows():
        stmt_items.append({
            'label': row['statement'],  # Full statement code like T01-S01
            'rt': row['rt'],
            'type': 'statement'
        })
    
    # Get individual articles (show full codes)
    for _, row in topic_arts.iterrows():
        art_items.append({
            'label': row['article'],  # Full article code like T01A
            'rt': row['rt'],
            'type': 'article'
        })
    
    # Sort items for better visualisation
    stmt_items = sorted(stmt_items, key=lambda x: x['label'])
    art_items = sorted(art_items, key=lambda x: x['label'])
    
    # Combine all items
    all_items = stmt_items + art_items
    
    if not all_items:
        ax.text(0.5, 0.5, f'No data for {topic}', ha='center', va='center')
        ax.set_title(f'Topic {topic}')
        return
    
    # Create bar plot
    positions = np.arange(len(all_items))
    colors = ['blue' if item['type'] == 'statement' else 'red' for item in all_items]
    labels = [item['label'] for item in all_items]
    rts = [item['rt'] for item in all_items]
    
    # Plot bars
    bars = ax.bar(positions, rts, color=colors, alpha=0.7)
    
    # Add dividing line between statements and articles if both exist
    if stmt_items and art_items:
        ax.axvline(x=len(stmt_items) - 0.5, color='gray', linestyle='--', alpha=0.5)
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Statements'),
                          Patch(facecolor='red', alpha=0.7, label='Articles')]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    # Set labels
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('RT (s)')
    ax.set_title(f'Topic {topic}: Individual RTs')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean lines
    if stmt_items:
        stmt_mean = np.mean([item['rt'] for item in stmt_items])
        ax.axhline(y=stmt_mean, color='blue', linestyle=':', alpha=0.5, linewidth=1)
    if art_items:
        art_mean = np.mean([item['rt'] for item in art_items])
        ax.axhline(y=art_mean, color='red', linestyle=':', alpha=0.5, linewidth=1)

def plot_summary_statistics(ax, statements, articles, bias_metrics):
    """Plot summary statistics panel"""
    ax.axis('off')
    
    # Calculate statistics
    n_statements = len(statements)
    n_articles = len(articles)
    n_valid_stmts = sum(1 for s in statements if s.attention_check == 'PASS')
    n_valid_arts = sum(1 for a in articles if a.attention_check == 'PASS')
    
    # RT statistics (only for valid responses)
    valid_stmt_rts = [s.reaction_time for s in statements if s.attention_check == 'PASS']
    valid_art_rts = [a.reading_time for a in articles if a.attention_check == 'PASS']
    
    stmt_rt_mean = np.mean(valid_stmt_rts) if valid_stmt_rts else 0
    stmt_rt_std = np.std(valid_stmt_rts) if valid_stmt_rts else 0
    art_rt_mean = np.mean(valid_art_rts) if valid_art_rts else 0
    art_rt_std = np.std(valid_art_rts) if valid_art_rts else 0
    
    # Create summary text
    summary = "ANALYSIS SUMMARY\n" + "="*30 + "\n\n"
    summary += f"Data Points:\n"
    summary += f"  Statements: {n_valid_stmts}/{n_statements} valid\n"
    summary += f"  Articles: {n_valid_arts}/{n_articles} valid\n"
    summary += f"  Attention Pass Rate: {bias_metrics['attention_pass_rate']:.1%}\n\n"
    
    summary += f"Reaction Times:\n"
    summary += f"  Statements: {stmt_rt_mean:.2f}±{stmt_rt_std:.2f}s\n"
    summary += f"  Articles: {art_rt_mean:.2f}±{art_rt_std:.2f}s\n\n"
    
    summary += f"Bias Analysis:\n"
    summary += f"  Valid Pairs: {bias_metrics['n_valid_pairs']}\n"
    summary += f"  Confirmation: {bias_metrics['confirmation_bias_rate']:.1%}\n"
    summary += f"  Disconfirmation: {bias_metrics['disconfirmation_seeking_rate']:.1%}\n"
    summary += f"  Weighted Bias Score: {bias_metrics['weighted_mean_bias_score']:.3f}\n"  # NEW
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax.set_title('Summary Statistics')

def plot_attention_performance(ax, stmt_df, art_df):
    """Plot attention check performance"""
    # Count attention check results
    stmt_counts = stmt_df['attention'].value_counts()
    art_counts = art_df['attention'].value_counts()
    
    # Prepare data
    categories = ['Passed', 'Failed']
    stmt_vals = [
        stmt_counts.get('PASS', 0),
        stmt_counts.get('FAIL', 0)
    ]
    art_vals = [
        art_counts.get('PASS', 0),
        art_counts.get('FAIL', 0)
    ]
    
    # Plot grouped bars
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, stmt_vals, width, label='Statements', color='blue', alpha=0.7)
    ax.bar(x + width/2, art_vals, width, label='Articles', color='red', alpha=0.7)
    
    ax.set_ylabel('Count')
    ax.set_title('Attention Check Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

def plot_rt_distributions(ax, stmt_df, art_df):
    """Plot RT distribution histograms"""
    # Create overlapping histograms
    if not stmt_df.empty:
        ax.hist(stmt_df['rt'], bins=20, alpha=0.5, label='Statements', color='blue', edgecolor='black')
    if not art_df.empty:
        ax.hist(art_df['rt'], bins=20, alpha=0.5, label='Articles', color='red', edgecolor='black')
    
    ax.set_xlabel('Reaction Time (s)')
    ax.set_ylabel('Frequency')
    ax.set_title('RT Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

# ============================================================================
# VISUALISATION FUNCTIONS - UPDATED WITH SPLIT FIGURES
# ============================================================================

def create_comprehensive_visualizations(statements: List[StatementResponse],
                                       articles: List[ArticleResponse],
                                       bias_metrics: Dict,
                                       output_dir: str):
    """Create all requested visualisations in two separate figures"""
    
    # Convert to DataFrames for easier analysis
    stmt_df = pd.DataFrame([{
        'topic': s.topic_code,
        'statement': s.statement_code,
        'agreement': s.agreement,
        'rt': s.reaction_time,
        'attention': s.attention_check
    } for s in statements])
    
    art_df = pd.DataFrame([{
        'article': a.article_code,
        'topic': a.topic_code,
        'agreement': a.agreement,
        'rt': a.reading_time,
        'attention': a.attention_check
    } for a in articles])
    
    # Filter for valid attention checks
    stmt_valid = stmt_df[stmt_df['attention'] == 'PASS']
    art_valid = art_df[art_df['attention'] == 'PASS']
    
    # ============================================
    # FIGURE 1: Summary Analysis Figure
    # ============================================
    fig1 = plt.figure(figsize=(20, 12))
    gs1 = fig1.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # 1. Response Distribution Comparison by Phase
    ax1 = fig1.add_subplot(gs1[0, 0])
    plot_response_distribution_by_phase(ax1, stmt_valid, art_valid)
    
    # 2. RT by Agreement Level Comparison
    ax2 = fig1.add_subplot(gs1[0, 1])
    plot_rt_by_agreement_comparison(ax2, stmt_valid, art_valid)
    
    # 3. Bias Distribution
    ax3 = fig1.add_subplot(gs1[0, 2])
    plot_bias_distribution(ax3, bias_metrics)
    
    # 4. Summary Statistics Text Panel
    ax4 = fig1.add_subplot(gs1[0, 3])
    plot_summary_statistics(ax4, statements, articles, bias_metrics)
    
    # 5. Average RT by Topic (spans full width of row 2)
    ax5 = fig1.add_subplot(gs1[1, :])
    plot_rt_by_topic(ax5, stmt_valid, art_valid)
    
    # 6. RT Distribution Histograms (bottom row, left side)
    ax6 = fig1.add_subplot(gs1[2, :2])  # Spans 2 columns in bottom row
    plot_rt_distributions(ax6, stmt_valid, art_valid)
    
    # 7. Attention Check Performance (bottom row, right side)
    ax7 = fig1.add_subplot(gs1[2, 2:])  # Spans 2 columns in bottom row
    plot_attention_performance(ax7, stmt_df, art_df)
    
    plt.suptitle('BCI Confirmation Bias Analysis - Summary', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'bci_analysis_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================
    # FIGURE 2: Individual Topic RT Analysis
    # ============================================
    fig2 = plt.figure(figsize=(20, 15))
    gs2 = fig2.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # Define all expected topics (12 topics used in experiment)
    all_topics = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 
                  'T15', 'T16', 'T17', 'T18', 'T20']
    
    # Get topics that have data
    topics_with_data = set(stmt_valid['topic'].unique()) | set(art_valid['topic'].unique())
    
    # Plot all 12 topics (3 rows x 4 columns)
    for i, topic in enumerate(all_topics):
        row = i // 4  # Rows 0, 1, 2
        col = i % 4   # Columns 0-3
        ax = fig2.add_subplot(gs2[row, col])
        
        # Check if this topic has data
        if topic in topics_with_data:
            plot_individual_rts_for_topic(ax, topic, stmt_valid, art_valid)
        else:
            # Empty plot for topics without data
            ax.text(0.5, 0.5, f'Topic {topic}\n(No data)', ha='center', va='center', 
                   fontsize=10, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Topic {topic}: Individual RTs')
    
    plt.suptitle('BCI Confirmation Bias Analysis - Individual Topic RTs', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'bci_analysis_individual_rts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualisations to {output_dir}")
    print(f"    - bci_analysis_summary.png")
    print(f"    - bci_analysis_individual_rts.png")

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def run_analysis(xdf_file: str = None, json_file: str = None, output_dir: str = './bci_output/'):
    """Run complete BCI analysis pipeline"""
    
    print("="*60)
    print("BCI CONFIRMATION BIAS ANALYSIS PIPELINE")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from available sources
    all_statements = []
    all_articles = []
    
    if xdf_file:
        print(f"\nLoading XDF: {xdf_file}")
        reconstructor = XDFReconstructor(xdf_file)
        xdf_statements, xdf_articles = reconstructor.reconstruct_responses()
        all_statements.extend(xdf_statements)
        all_articles.extend(xdf_articles)
        print(f"  Found {len(xdf_statements)} statements, {len(xdf_articles)} articles")
        
        # Print RT statistics
        if xdf_statements:
            rts = [s.reaction_time for s in xdf_statements if s.attention_check == 'PASS']
            if rts:
                print(f"  Statement RTs: {np.mean(rts):.2f}±{np.std(rts):.2f}s (n={len(rts)})")
        if xdf_articles:
            rts = [a.reading_time for a in xdf_articles if a.attention_check == 'PASS']
            if rts:
                print(f"  Article RTs: {np.mean(rts):.2f}±{np.std(rts):.2f}s (n={len(rts)})")
    
    if json_file:
        print(f"\nLoading JSON: {json_file}")
        json_statements, json_articles = load_json_data(json_file)
        
        # Only add if not already loaded from XDF
        if not all_statements:
            all_statements.extend(json_statements)
        if not all_articles:
            all_articles.extend(json_articles)
        
        print(f"  Found {len(json_statements)} statements, {len(json_articles)} articles")
    
    if not all_statements and not all_articles:
        print("\nERROR: No data loaded!")
        return
    
    # CORRECTED v4.0: Don't filter - pass ALL responses to calculate_bias_metrics
    # Attention checks are tracked WITHIN the function for quality reporting
    print(f"\nData loaded: {len(all_statements)} statements, {len(all_articles)} articles")

    # Calculate bias metrics using ALL data (no filtering)
    print("\nCalculating bias metrics...")
    bias_metrics = calculate_bias_metrics(all_statements, all_articles)
    
    print(f"  Total articles: {bias_metrics['n_total_articles']}")
    print(f"  Classifiable: {bias_metrics['n_classifiable']}")
    print(f"  Confirmation bias rate: {bias_metrics['confirmation_bias_rate']:.1%}")
    print(f"  Disconfirmation rate: {bias_metrics['disconfirmation_seeking_rate']:.1%}")
    print(f"  Classifiable rate: {bias_metrics['classifiable_rate']:.1%}")
    print(f"  Weighted mean bias score: {bias_metrics['weighted_mean_bias_score']:.3f}")
    print(f"  Attention pass rate: {bias_metrics['attention_pass_rate']:.1%}")
    
    # Generate visualisations
    print("\nGenerating visualisations...")
    create_comprehensive_visualizations(all_statements, all_articles, bias_metrics, output_dir)
    
    # Export data
    print("\nExporting data...")
    
    # Export statements
    stmt_data = []
    for s in all_statements:
        stmt_data.append({
            'topic': s.topic_code,
            'statement': s.statement_code,
            'agreement': s.agreement,
            'reaction_time': s.reaction_time,
            'attention_check': s.attention_check,
            'valid': 1 if s.attention_check == 'PASS' else 0
        })
    stmt_df = pd.DataFrame(stmt_data)
    stmt_df.to_csv(os.path.join(output_dir, 'statements.csv'), index=False)
    
    # Export articles with bias analysis - FIXED SECTION
    art_data = []
    bias_lookup = {b['article_code']: b for b in bias_metrics['detailed_analyses']}
    
    for a in all_articles:
        # Get article info from mappings
        article_info = get_article_info(a.article_code)
        
        row = {
            'article': a.article_code,
            'topic': a.topic_code,
            'agreement': a.agreement,
            'reading_time': a.reading_time,
            'attention_check': a.attention_check,
            'valid': 1 if a.attention_check == 'PASS' else 0,
            # Always include article_type from mapping
            'article_type': article_info.get('type', 'unknown')
        }
        
        # Add bias analysis info if available
        if a.article_code in bias_lookup:
            bias_info = bias_lookup[a.article_code]
            row.update({
                'bias_type': bias_info['bias_type'],
                'original_agreement': bias_info.get('original_agreement'),
                'primary_relationship': bias_info.get('primary_relationship', ''),  # NEW
                'relationship_weight': bias_info.get('relationship_weight', 0),      # NEW
                'weighted_bias_score': bias_info.get('weighted_bias_score', 0)       # NEW
            })
        else:
            # Set defaults for articles not in bias analysis
            row.update({
                'bias_type': None,
                'original_agreement': None,
                'primary_relationship': None,    # NEW
                'relationship_weight': None,      # NEW
                'weighted_bias_score': None       # NEW
            })
        
        art_data.append(row)
    
    art_df = pd.DataFrame(art_data)
    art_df.to_csv(os.path.join(output_dir, 'articles_with_bias.csv'), index=False)
    
    # Export summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_source': {
            'xdf_file': xdf_file,
            'json_file': json_file
        },
        'sample_size': {
            'total_statements': len(all_statements),
            'valid_statements': sum(1 for s in all_statements if s.attention_check == 'PASS'),
            'total_articles': len(all_articles),
            'valid_articles': sum(1 for a in all_articles if a.attention_check == 'PASS')
        },
        'bias_metrics': bias_metrics,
        'rt_summary': {
            'statements': {
                'mean': np.mean([s.reaction_time for s in all_statements if s.attention_check == 'PASS']),
                'std': np.std([s.reaction_time for s in all_statements if s.attention_check == 'PASS'])
            } if all_statements else {},
            'articles': {
                'mean': np.mean([a.reading_time for a in all_articles if a.attention_check == 'PASS']),
                'std': np.std([a.reading_time for a in all_articles if a.attention_check == 'PASS'])
            } if all_articles else {}
        }
    }
    
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"  ✓ Exported all data to {output_dir}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Streamlined BCI Analysis Pipeline')
    parser.add_argument('--xdf', help='XDF file path')
    parser.add_argument('--json', help='JSON file path')
    parser.add_argument('--output', default='./bci_output/patched/', help='Output directory')
    
    args = parser.parse_args()
    

    
    ## Use test files if no arguments provided
    #if not args.xdf and not args.json:
    #    if os.path.exists('test_session6.xdf'):
    #        args.xdf = 'test_session6.xdf'
    #    if os.path.exists('participant_030303_20250918_151935.json'):
    #        args.json = 'participant_030303_20250918_151935.json'
    
    run_analysis(args.xdf, args.json, args.output)