#!/usr/bin/env python3
"""
INDIVIDUAL ARTICLE ANALYSIS ACROSS 8 DECISION STAGES
===============================================================================
Analyzes each article separately through the complete decision-making cascade:
1. TOPIC_SELECT - Interest selection
2. ARTICLE_SELECTOR_START - List appears  
3. ARTICLE_PREVIEW - Headline hover
4. ARTICLE_SELECT - Choice made (PRIMARY BIAS MOMENT)
5. BIOSYNC_STIMULUS_ONSET_ARTICLE - Reading onset sync
6. ARTICLE_READ_START - Content processing begins
7. Sustained reading (2-10s window)
8. ARTICLE_RATING - Final response

Outputs structured data for visualization script.

Author: Jason Stewart (25182902)
Ethics: ETH23-7909
Date: November 3, 2025
Version: 1.0
===============================================================================
"""

import numpy as np
import pandas as pd
import mne
from scipy import stats
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Handle different MNE versions for PSD computation
try:
    from mne.time_frequency import psd_array_welch
    USE_NEW_MNE = True
    print("✓ Using MNE 1.0+ API (psd_array_welch)")
except ImportError:
    try:
        from mne.time_frequency import psd_welch
        USE_NEW_MNE = False
        print("✓ Using MNE 0.23+ API (psd_welch)")
    except ImportError:
        USE_NEW_MNE = None
        print("⚠ Using scipy.signal.welch (legacy fallback)")

# ===================================================================================
# 8 ARTICLE DECISION STAGES CONFIGURATION
# ===================================================================================

ARTICLE_STAGES = {
    'topic_select': {
        'name': 'Topic Selection',
        'marker': 'TOPIC_SELECT',
        'tmin': -0.5,
        'tmax': 2.0,
        'baseline': (-0.5, 0),
        'theta_window': (0, 1.5),
        'description': 'Interest selection - choosing topic area',
        'phase': 'selection'
    },
    'selector_start': {
        'name': 'Article List Presentation',
        'marker': 'ARTICLE_SELECTOR_START',
        'tmin': 0,
        'tmax': 3.0,
        'baseline': (0, 0.2),
        'theta_window': (0.2, 2.5),
        'description': 'Article list appears - decision preparation',
        'phase': 'selection'
    },
    'article_preview': {
        'name': 'Article Preview (Hover)',
        'marker': 'ARTICLE_PREVIEW',
        'tmin': -0.2,
        'tmax': 2.0,
        'baseline': (-0.2, 0),
        'theta_window': (0, 1.5),
        'description': 'Headline consideration - selective attention',
        'phase': 'selection'
    },
    'article_select': {
        'name': 'Article Selection',
        'marker': 'ARTICLE_SELECT',
        'tmin': -0.5,
        'tmax': 2.0,
        'baseline': (-0.5, 0),
        'theta_window': (0, 1.5),
        'description': 'Choice commitment - PRIMARY BIAS MOMENT',
        'phase': 'selection'
    },
    'article_biosync': {
        'name': 'Reading Onset (BioSync)',
        'marker': 'BIOSYNC_STIMULUS_ONSET_ARTICLE',
        'tmin': 0,
        'tmax': 2.0,
        'baseline': (0, 0.25),
        'theta_window': (0.25, 1.25),
        'description': 'Reading begins - initial processing',
        'phase': 'reading'
    },
    'reading_start': {
        'name': 'Reading Start',
        'marker': 'ARTICLE_READ_START',
        'tmin': 0,
        'tmax': 2.0,
        'baseline': (0, 0.2),
        'theta_window': (0.25, 1.5),
        'description': 'Content processing onset',
        'phase': 'reading'
    },
    'reading_sustained': {
        'name': 'Sustained Reading',
        'marker': 'ARTICLE_READ_START',
        'tmin': 2.0,
        'tmax': 10.0,
        'baseline': (2.0, 2.5),
        'theta_window': (2.5, 8.0),
        'description': 'Continued content processing',
        'phase': 'reading'
    },
    'article_rating': {
        'name': 'Article Rating',
        'marker': 'ARTICLE_RATING',
        'tmin': -0.5,
        'tmax': 2.0,
        'baseline': (-0.5, 0),
        'theta_window': (0, 1.5),
        'description': 'Final response - belief update',
        'phase': 'response'
    }
}


class IndividualArticleAnalyzer:
    """
    Analyzes each article individually across all 8 decision stages
    """
    
    def __init__(self,
                 epochs_file: str,
                 metadata_file: str,
                 articles_file: str,
                 output_dir: str = './bci_output/individual_article_analysis'):
        
        self.epochs_file = epochs_file
        self.metadata_file = metadata_file
        self.articles_file = articles_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Data containers
        self.epochs = None
        self.epochs_metadata = None
        self.articles = None
        
        # Results
        self.individual_results = []
        
        print(f"\n{'='*80}")
        print(f"INDIVIDUAL ARTICLE ANALYSIS - 8 DECISION STAGES")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"MNE version: {mne.__version__}")
        print(f"{'='*80}\n")
    
    def load_data(self):
        """Load EEG epochs, metadata, and article classifications"""
        print(f"{'='*80}")
        print("LOADING DATA")
        print(f"{'='*80}\n")
        
        # Load epochs
        print("[1/3] Loading EEG epochs...")
        self.epochs = mne.read_epochs(self.epochs_file, verbose=False)
        print(f"  ✓ Loaded {len(self.epochs)} epochs (articles)")
        
        # Load metadata
        print("\n[2/3] Loading epochs metadata...")
        self.epochs_metadata = pd.read_csv(self.metadata_file)
        print(f"  ✓ Loaded {len(self.epochs_metadata)} metadata records")
        
        # Load article classifications
        print("\n[3/3] Loading article classifications...")
        self.articles = pd.read_csv(self.articles_file)
        print(f"  ✓ Loaded {len(self.articles)} articles")
        
        # Show bias type distribution
        bias_counts = self.articles['bias_type'].value_counts()
        print(f"\n  Bias type distribution:")
        for bias_type, count in bias_counts.items():
            print(f"    {bias_type}: {count}")
        
        # Show attention check distribution
        if 'valid' in self.articles.columns:
            n_pass = self.articles['valid'].sum()
            n_fail = len(self.articles) - n_pass
            print(f"\n  Attention checks:")
            print(f"    PASS: {n_pass}")
            print(f"    FAIL: {n_fail}")
        
        print(f"\n{'='*80}")
        print("DATA LOADING COMPLETE")
        print(f"{'='*80}\n")
    
    def _compute_psd_compatible(self, epochs_data, sfreq, fmin, fmax):
        """Compute PSD with version compatibility"""
        n_epochs, n_channels, n_times = epochs_data.shape
        
        desired_window_samples = min(int(sfreq * 2), n_times)
        n_fft = min(2 ** int(np.log2(desired_window_samples)), n_times)
        n_per_seg = n_fft
        n_overlap = n_fft // 2
        
        if n_overlap >= n_per_seg:
            n_overlap = n_per_seg // 2
        
        if USE_NEW_MNE:
            psds, freqs = psd_array_welch(
                epochs_data, sfreq=sfreq, fmin=fmin, fmax=fmax,
                n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, verbose=False
            )
        else:
            from scipy.signal import welch
            psds_list = []
            for epoch_data in epochs_data:
                epoch_psds = []
                for channel_data in epoch_data:
                    freqs, psd = welch(
                        channel_data, fs=sfreq, nperseg=n_per_seg,
                        noverlap=n_overlap, nfft=n_fft
                    )
                    freq_mask = (freqs >= fmin) & (freqs <= fmax)
                    epoch_psds.append(psd[freq_mask])
                psds_list.append(epoch_psds)
            psds = np.array(psds_list)
            freqs = freqs[freq_mask]
        
        return psds, freqs
    
    def analyze_individual_articles(self):
        """
        Analyze each article through all 8 stages
        """
        print(f"{'='*80}")
        print("ANALYZING INDIVIDUAL ARTICLES ACROSS 8 STAGES")
        print(f"{'='*80}\n")
        
        # Get frontal channels for theta analysis
        all_channels = self.epochs.ch_names
        frontal_channels = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8']
        available_frontal = [ch for ch in frontal_channels if ch in all_channels]
        
        print(f"Channels: {len(all_channels)} total, {len(available_frontal)} frontal\n")
        
        # Merge metadata with article info
        merged_data = self.epochs_metadata.merge(
            self.articles[['article', 'bias_type', 'article_type', 'valid', 'attention_check']],
            left_on='article_code',
            right_on='article',
            how='left'
        )
        
        # Process each article
        for article_idx in range(len(merged_data)):
            article_info = merged_data.iloc[article_idx]
            article_code = article_info['article_code']
            bias_type = article_info['bias_type']
            attention_pass = article_info['valid'] == 1
            
            print(f"{'='*80}")
            print(f"ARTICLE: {article_code}")
            print(f"{'='*80}")
            print(f"  Bias type: {bias_type}")
            print(f"  Attention check: {'PASS' if attention_pass else 'FAIL'}")
            print()
            
            # Convert numpy/pandas types to native Python types for JSON serialization
            article_result = {
                'article_code': str(article_code),
                'article_idx': int(article_idx),
                'bias_type': str(bias_type) if pd.notna(bias_type) else None,
                'attention_pass': bool(attention_pass),
                'stages': {}
            }
            
            # Analyze each stage for this article
            for stage_id, stage_config in ARTICLE_STAGES.items():
                print(f"  Stage: {stage_config['name']}")
                
                try:
                    # Get epoch for this article
                    epoch = self.epochs[[article_idx]].copy()
                    
                    # Crop to stage time window
                    tmin = stage_config['tmin']
                    tmax = stage_config['tmax']
                    epoch_cropped = epoch.copy().crop(tmin=tmin, tmax=tmax)
                    
                    # Apply baseline
                    baseline = stage_config['baseline']
                    epoch_cropped.apply_baseline(baseline)
                    
                    # Extract ERP (average across this single epoch)
                    erp_data = epoch_cropped.get_data()[0]  # Shape: (n_channels, n_times)
                    times = epoch_cropped.times
                    
                    # Get frontal ERP
                    epoch_frontal = epoch.copy().crop(tmin=tmin, tmax=tmax)
                    epoch_frontal.pick_channels(available_frontal, verbose=False)
                    epoch_frontal.apply_baseline(baseline)
                    erp_frontal = epoch_frontal.get_data()[0]  # Shape: (n_frontal, n_times)
                    
                    # Compute theta power
                    theta_tmin, theta_tmax = stage_config['theta_window']
                    epoch_theta = epoch.copy().crop(tmin=theta_tmin, tmax=theta_tmax)
                    
                    # All channels theta
                    epochs_data_all = epoch_theta.get_data()
                    sfreq = epoch_theta.info['sfreq']
                    psds_all, freqs = self._compute_psd_compatible(
                        epochs_data_all, sfreq=sfreq, fmin=4.0, fmax=8.0
                    )
                    theta_all = psds_all.mean(axis=(1, 2))[0]  # Single value for this epoch
                    
                    # Frontal theta
                    epoch_theta_frontal = epoch.copy().crop(tmin=theta_tmin, tmax=theta_tmax)
                    epoch_theta_frontal.pick_channels(available_frontal, verbose=False)
                    epochs_data_frontal = epoch_theta_frontal.get_data()
                    psds_frontal, _ = self._compute_psd_compatible(
                        epochs_data_frontal, sfreq=sfreq, fmin=4.0, fmax=8.0
                    )
                    theta_frontal = psds_frontal.mean(axis=(1, 2))[0]
                    
                    # Store results
                    article_result['stages'][stage_id] = {
                        'stage_name': stage_config['name'],
                        'phase': stage_config['phase'],
                        'erp_times': times.tolist(),
                        'erp_all_channels': erp_data.mean(axis=0).tolist(),  # Average across all channels
                        'erp_frontal': erp_frontal.mean(axis=0).tolist(),  # Average across frontal
                        'theta_all': float(theta_all),
                        'theta_frontal': float(theta_frontal),
                        'n_channels_all': len(all_channels),
                        'n_channels_frontal': len(available_frontal)
                    }
                    
                    print(f"    ✓ Theta (all): {theta_all:.4e}, Theta (frontal): {theta_frontal:.4e}")
                    
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    article_result['stages'][stage_id] = {
                        'stage_name': stage_config['name'],
                        'phase': stage_config['phase'],
                        'error': str(e)
                    }
            
            self.individual_results.append(article_result)
            print()
        
        print(f"{'='*80}")
        print(f"✓ INDIVIDUAL ARTICLE ANALYSIS COMPLETE")
        print(f"{'='*80}\n")
    
    def save_results(self):
        """Save analysis results in multiple formats for visualization"""
        print(f"{'='*80}")
        print("SAVING RESULTS")
        print(f"{'='*80}\n")
        
        # 1. Save complete JSON
        json_path = self.output_dir / 'individual_article_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.individual_results, f, indent=2)
        print(f"✓ Saved complete results: {json_path}")
        
        # 2. Save theta power matrix (articles × stages)
        theta_data = []
        for result in self.individual_results:
            row = {
                'article_code': result['article_code'],
                'bias_type': result['bias_type'],
                'attention_pass': result['attention_pass']
            }
            for stage_id, stage_config in ARTICLE_STAGES.items():
                if stage_id in result['stages'] and 'theta_frontal' in result['stages'][stage_id]:
                    row[f'{stage_id}_theta'] = result['stages'][stage_id]['theta_frontal']
                else:
                    row[f'{stage_id}_theta'] = np.nan
            theta_data.append(row)
        
        theta_df = pd.DataFrame(theta_data)
        theta_csv_path = self.output_dir / 'theta_power_matrix.csv'
        theta_df.to_csv(theta_csv_path, index=False)
        print(f"✓ Saved theta power matrix: {theta_csv_path}")
        
        # 3. Save ERP summary (for quick access)
        erp_summary = []
        for result in self.individual_results:
            for stage_id, stage_data in result['stages'].items():
                if 'erp_frontal' in stage_data:
                    erp_summary.append({
                        'article_code': result['article_code'],
                        'bias_type': result['bias_type'],
                        'attention_pass': result['attention_pass'],
                        'stage_id': stage_id,
                        'stage_name': stage_data['stage_name'],
                        'phase': stage_data['phase'],
                        'erp_file': f"{result['article_code']}_{stage_id}_erp.npy"
                    })
                    
                    # Save individual ERP arrays
                    erp_array = np.array(stage_data['erp_frontal'])
                    times_array = np.array(stage_data['erp_times'])
                    erp_npy_path = self.output_dir / f"{result['article_code']}_{stage_id}_erp.npy"
                    np.save(erp_npy_path, {'times': times_array, 'erp': erp_array})
        
        erp_summary_df = pd.DataFrame(erp_summary)
        erp_summary_path = self.output_dir / 'erp_summary.csv'
        erp_summary_df.to_csv(erp_summary_path, index=False)
        print(f"✓ Saved ERP summary: {erp_summary_path}")
        
        # 4. Save article metadata
        metadata = []
        for result in self.individual_results:
            metadata.append({
                'article_code': result['article_code'],
                'article_idx': result['article_idx'],
                'bias_type': result['bias_type'],
                'attention_pass': result['attention_pass']
            })
        
        metadata_df = pd.DataFrame(metadata)
        metadata_path = self.output_dir / 'article_metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)
        print(f"✓ Saved article metadata: {metadata_path}")
        
        print(f"\n{'='*80}")
        print("RESULTS SAVED SUCCESSFULLY")
        print(f"{'='*80}\n")
    
    def run_analysis(self):
        """Execute complete individual article analysis"""
        self.load_data()
        self.analyze_individual_articles()
        self.save_results()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Files generated:")
        print(f"  - individual_article_results.json (complete data)")
        print(f"  - theta_power_matrix.csv (articles × stages)")
        print(f"  - erp_summary.csv (ERP metadata)")
        print(f"  - article_metadata.csv (article info)")
        print(f"  - *_erp.npy (individual ERP arrays)")
        print(f"\nReady for visualization script!")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Individual Article Analysis - 8 Decision Stages'
    )
    
    parser.add_argument('--epochs', required=True,
                       help='Path to epochs file (.fif)')
    parser.add_argument('--metadata', required=True,
                       help='Path to metadata CSV')
    parser.add_argument('--articles', required=True,
                       help='Path to articles with bias CSV')
    parser.add_argument('--output', '-o', 
                       default='./bci_output/individual_article_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = IndividualArticleAnalyzer(
        epochs_file=args.epochs,
        metadata_file=args.metadata,
        articles_file=args.articles,
        output_dir=args.output
    )
    
    analyzer.run_analysis()