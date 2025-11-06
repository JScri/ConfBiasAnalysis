#!/usr/bin/env python3
"""
EEG Preprocessing Pipeline for Confirmation Bias Study - NO ATTENTION FILTERING
VERSION 4.0 - November 2, 2025

CRITICAL PHILOSOPHY:
- Article selection happens BEFORE attention checks
- Therefore, ALL epochs are kept regardless of attention check status
- Attention check status is stored as METADATA ONLY
- Filtering decisions are made during ANALYSIS, not preprocessing

KEY CHANGES FROM V3.1:
1. Removed 'valid' flag - replaced with 'attention_passed' for clarity
2. All epochs kept regardless of attention checks
3. Added separate epoch creation methods for different analysis stages
4. Explicit documentation that filtering happens at analysis stage

Author: Jason Stewart
Date: November 2, 2025
Ethics: ETH23-7909
"""

import os
import numpy as np
import pandas as pd
import mne
import pyxdf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """
    Preprocesses EEG data and creates epochs WITHOUT attention check filtering.
    
    DESIGN PRINCIPLE:
    Confirmation bias occurs at the moment of article selection, which happens
    BEFORE participants read content and BEFORE attention checks. Therefore,
    selection-locked epochs must include ALL trials to properly measure bias.
    
    Attention check status is preserved as metadata for optional filtering
    during analysis stages where engagement matters (reading, post-rating).
    """
    
    # Hardware specifications
    SAMPLING_RATE = 125.0
    ADC_TO_UV_SCALE = 0.022351744455307063
    DISCONNECTED_CHANNEL = 14  # Ch14 is disconnected
    
    # Channel mapping (10-20 system)
    CHANNEL_NAMES = [
        'Fp1', 'Fp2', 'C3', 'C4', 'Fz', 'Cz', 'O1', 'O2',
        'F7', 'F8', 'F3', 'F4', 'Pz', 'DISCONNECTED', 'P3', 'P4'
    ]
    
    def __init__(self, xdf_file: str, output_dir: str):
        """
        Initialize preprocessor
        
        Args:
            xdf_file: Path to XDF file from LabRecorder
            output_dir: Directory for preprocessed outputs
        """
        self.xdf_file = xdf_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Data containers
        self.raw = None
        self.eeg_timestamps = None
        self.markers_df = None
        self.attention_check_map = None
        
        print(f"\n{'='*80}")
        print(f"EEG PREPROCESSING PIPELINE - V4.0 (NO ATTENTION FILTERING)")
        print(f"{'='*80}")
        print(f"Input: {xdf_file}")
        print(f"Output: {output_dir}")
        print(f"\n⚠️  ATTENTION FILTERING DISABLED AT PREPROCESSING STAGE")
        print(f"   All epochs kept with attention status as metadata only")
        print(f"{'='*80}\n")
    
    def load_xdf(self):
        """Load XDF file and extract EEG + marker streams"""
        print(f"{'='*80}")
        print("STEP 1: LOADING XDF DATA")
        print(f"{'='*80}")
        
        streams, header = pyxdf.load_xdf(self.xdf_file)
        print(f"Found {len(streams)} streams")
        
        # Find EEG stream
        eeg_stream = None
        marker_stream = None
        
        for stream in streams:
            stream_type = stream['info']['type'][0].lower()
            stream_name = stream['info']['name'][0]
            
            if 'eeg' in stream_type:
                eeg_stream = stream
            elif 'marker' in stream_type or 'marker' in stream_name.lower():
                marker_stream = stream
        
        if not eeg_stream:
            raise ValueError("No EEG stream found in XDF file")
        if not marker_stream:
            raise ValueError("No marker stream found in XDF file")
        
        # Extract EEG data (in ADC counts)
        data_counts = eeg_stream['time_series'].T
        self.eeg_timestamps = eeg_stream['time_stamps']
        
        print(f"✓ EEG data loaded: {data_counts.shape[0]} channels × {data_counts.shape[1]} samples")
        print(f"  Duration: {data_counts.shape[1] / self.SAMPLING_RATE:.1f} seconds")
        
        # Convert to microvolts then to volts (for MNE)
        data_uv = data_counts * self.ADC_TO_UV_SCALE
        data_volts = data_uv * 1e-6
        
        print(f"  Signal range: {np.min(data_uv):.1f} to {np.max(data_uv):.1f} µV")
        print(f"  Mean RMS: {np.mean(np.std(data_uv, axis=1)):.1f} µV")
        
        # Create MNE Raw object
        channel_names = [ch for ch in self.CHANNEL_NAMES]
        
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=self.SAMPLING_RATE,
            ch_types=['eeg'] * len(channel_names)
        )
        
        self.raw = mne.io.RawArray(data_volts, info, verbose=False)
        
        # Set montage
        montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(montage, on_missing='ignore', verbose=False)
        
        # Drop disconnected channel
        self.raw.drop_channels(['DISCONNECTED'])
        print(f"✓ Dropped DISCONNECTED channel, remaining: {len(self.raw.ch_names)} channels")
        
        # Process markers
        self._process_markers(marker_stream)
        
        # Build attention check map (metadata only - NOT for filtering)
        self._build_attention_check_map()
        
        return self.raw
    
    def _process_markers(self, marker_stream):
        """Convert markers to structured DataFrame"""
        print(f"\n{'='*80}")
        print("PROCESSING EVENT MARKERS")
        print(f"{'='*80}")
        
        markers = marker_stream['time_series']
        marker_times = marker_stream['time_stamps']
        
        # Parse each marker
        marker_data = []
        for i, (marker, timestamp) in enumerate(zip(markers, marker_times)):
            marker_str = marker[0] if isinstance(marker, (list, np.ndarray)) else str(marker)
            
            # Convert to EEG sample index
            time_diffs = np.abs(self.eeg_timestamps - timestamp)
            sample_idx = np.argmin(time_diffs)
            
            marker_data.append({
                'index': i,
                'sample': sample_idx,
                'timestamp': timestamp,
                'marker': marker_str,
                'category': self._categorize_marker(marker_str)
            })
        
        self.markers_df = pd.DataFrame(marker_data)
        
        # Count by category
        category_counts = self.markers_df['category'].value_counts()
        print(f"✓ Processed {len(self.markers_df)} markers:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
    
    def _categorize_marker(self, marker_str: str) -> str:
        """Categorize marker type"""
        marker_upper = marker_str.upper()
        
        if 'STATEMENT_PRESENTED' in marker_upper:
            return 'STATEMENT_START'
        elif 'STATEMENT_RESPONSE' in marker_upper:
            return 'STATEMENT_RESPONSE'
        elif 'ATTENTION_CHECK_PRESENTED' in marker_upper:
            return 'ATTN_START_STMT'
        elif 'ATTENTION_CHECK_RESPONSE' in marker_upper and 'PHASE2' not in marker_upper:
            return 'ATTN_RESPONSE_STMT'
        elif 'ATTENTION_CHECK_START_PHASE2' in marker_upper:
            return 'ATTN_START_ART'
        elif 'ATTENTION_CHECK_RESPONSE_PHASE2' in marker_upper:
            return 'ATTN_RESPONSE_ART'
        elif 'TOPIC_SELECT' in marker_upper:
            return 'TOPIC_SELECT'
        elif 'ARTICLE_READ_START' in marker_upper:
            return 'ARTICLE_START'
        elif 'ARTICLE_RATING' in marker_upper:
            return 'ARTICLE_RATING'
        else:
            return 'OTHER'
    
    def _extract_statement_code(self, marker: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        Extract question index, topic, and statement code from marker
        
        Args:
            marker: Marker string like "STATEMENT_PRESENTED_Q0_T15_S05"
        
        Returns:
            Tuple of (question_index, topic_code, statement_code)
        """
        parts = marker.split('_')
        
        q_idx = None
        topic_code = None
        statement_code = None
        
        for part in parts:
            if part.startswith('Q') and part[1:].isdigit():
                q_idx = int(part[1:])
            elif part.startswith('T') and len(part) >= 3:
                topic_code = part[:3]
            elif part.startswith('S') and len(part) == 3 and part[1:].isdigit():
                statement_code = part
        
        return q_idx, topic_code, statement_code
    
    def _build_attention_check_map(self):
        """
        Build map of attention check responses for ALL trials.
        
        IMPORTANT: This is for METADATA ONLY, not filtering.
        Attention checks occur AFTER article selection, so they cannot
        invalidate the selection decision itself.
        """
        print(f"\n{'='*80}")
        print("BUILDING ATTENTION CHECK MAP (METADATA ONLY)")
        print(f"{'='*80}")
        print("⚠️  This map is for metadata annotation, NOT for filtering epochs")
        
        self.attention_check_map = {
            'statements': {},
            'articles': {}
        }
        
        # Get statement attention check pairs
        stmt_starts = self.markers_df[self.markers_df['category'] == 'ATTN_START_STMT']
        stmt_responses = self.markers_df[self.markers_df['category'] == 'ATTN_RESPONSE_STMT']
        
        print(f"\nStatement attention checks:")
        print(f"  Starts: {len(stmt_starts)}")
        print(f"  Responses: {len(stmt_responses)}")
        
        for idx, start_row in stmt_starts.iterrows():
            marker = start_row['marker']
            q_idx, topic_code, stmt_code = self._extract_statement_code(marker)
            
            if topic_code and stmt_code:
                key = f"{topic_code}-{stmt_code}"
                
                # Find corresponding response
                next_responses = stmt_responses[stmt_responses['index'] > start_row['index']]
                
                if len(next_responses) > 0:
                    response_row = next_responses.iloc[0]
                    response_marker = response_row['marker']
                    
                    # Check response
                    if 'YES' in response_marker.upper():
                        status = 'PASS'
                    elif 'NO' in response_marker.upper():
                        status = 'PASS'
                    else:
                        status = 'FAIL'
                    
                    self.attention_check_map['statements'][key] = status
                else:
                    self.attention_check_map['statements'][key] = 'MISSING'
        
        # Get article attention check pairs
        art_starts = self.markers_df[self.markers_df['category'] == 'ATTN_START_ART']
        art_responses = self.markers_df[self.markers_df['category'] == 'ATTN_RESPONSE_ART']
        
        print(f"\nArticle attention checks:")
        print(f"  Starts: {len(art_starts)}")
        print(f"  Responses: {len(art_responses)}")
        
        for idx, start_row in art_starts.iterrows():
            marker = start_row['marker']
            parts = marker.split('_')
            
            article_code = None
            for part in parts:
                if part.startswith('T') and len(part) >= 4:
                    article_code = part
                    break
            
            if article_code:
                # Find corresponding response
                next_responses = art_responses[art_responses['index'] > start_row['index']]
                
                if len(next_responses) > 0:
                    response_row = next_responses.iloc[0]
                    response_marker = response_row['marker']
                    
                    # Check response
                    if 'YES' in response_marker.upper():
                        status = 'PASS'
                    elif 'NO' in response_marker.upper():
                        status = 'PASS'
                    else:
                        status = 'FAIL'
                    
                    self.attention_check_map['articles'][article_code] = status
                else:
                    self.attention_check_map['articles'][article_code] = 'MISSING'
        
        # Summary
        total_stmt_checks = len(self.attention_check_map['statements'])
        passed_stmt = sum(1 for v in self.attention_check_map['statements'].values() if v == 'PASS')
        
        total_art_checks = len(self.attention_check_map['articles'])
        passed_art = sum(1 for v in self.attention_check_map['articles'].values() if v == 'PASS')
        
        print(f"\n✓ Attention check map built:")
        print(f"  Statements: {passed_stmt}/{total_stmt_checks} passed")
        print(f"  Articles: {passed_art}/{total_art_checks} passed")
        print(f"\n⚠️  Remember: These are stored as METADATA, not used for filtering")
    
    def preprocess(self, highpass=0.5, lowpass=40.0, notch_freq=50.0):
        """Apply preprocessing: filtering and rereferencing"""
        print(f"\n{'='*80}")
        print("STEP 2: PREPROCESSING")
        print(f"{'='*80}")
        
        # Filter
        print(f"Applying filters:")
        print(f"  Highpass: {highpass} Hz")
        print(f"  Lowpass: {lowpass} Hz")
        print(f"  Notch: {notch_freq} Hz")
        
        self.raw.filter(highpass, lowpass, verbose=False)
        self.raw.notch_filter(notch_freq, verbose=False)
        
        # Rereference to average
        print(f"\nRereferencing to average")
        self.raw.set_eeg_reference('average', projection=True, verbose=False)
        self.raw.apply_proj(verbose=False)
        
        print(f"✓ Preprocessing complete")
    
    def create_statement_epochs(self, tmin=-0.5, tmax=2.0) -> Tuple[mne.Epochs, pd.DataFrame]:
        """
        Create epochs for statement presentation.
        
        FILTERING POLICY: NONE (Tier 1 - Never Filter)
        Rationale: Statement rating occurs before attention checks,
        so attention check status is irrelevant to the rating decision.
        
        All epochs kept with attention_check status as metadata.
        """
        print(f"\n{'='*80}")
        print("STEP 3: CREATING STATEMENT EPOCHS (NO ATTENTION FILTERING)")
        print(f"{'='*80}")
        print("⚠️  Tier 1 Analysis: All epochs kept regardless of attention checks")
        
        # Find statement markers
        stmt_starts = self.markers_df[self.markers_df['category'] == 'STATEMENT_START']
        stmt_responses = self.markers_df[self.markers_df['category'] == 'STATEMENT_RESPONSE']
        
        print(f"Found {len(stmt_starts)} statement presentations")
        print(f"Found {len(stmt_responses)} statement responses")
        
        # Build metadata for ALL statements
        epoch_metadata = []
        events_list = []
        
        for idx, start_row in stmt_starts.iterrows():
            marker = start_row['marker']
            q_idx, topic_code, stmt_code = self._extract_statement_code(marker)
            
            if not (topic_code and stmt_code):
                continue
            
            # Find corresponding response
            next_responses = stmt_responses[
                (stmt_responses['index'] > start_row['index']) &
                (stmt_responses['marker'].str.contains(topic_code)) &
                (stmt_responses['marker'].str.contains(stmt_code))
            ]
            
            if len(next_responses) == 0:
                continue
            
            response_row = next_responses.iloc[0]
            
            # Extract rating
            rating_parts = [p for p in response_row['marker'].split('_') 
                           if p.startswith('R') and len(p) == 2 and p[1].isdigit()]
            rating = int(rating_parts[0][1]) if rating_parts else 0
            
            # Look up attention check status (METADATA ONLY)
            key = f"{topic_code}-{stmt_code}"
            attention_status = self.attention_check_map['statements'].get(key, 'MISSING')
            
            # Calculate reaction time
            rt = response_row['timestamp'] - start_row['timestamp']
            
            epoch_metadata.append({
                'question_index': q_idx,
                'topic_code': topic_code,
                'statement_code': stmt_code,
                'statement_key': key,
                'rating': rating,
                'reaction_time': rt,
                'attention_check': attention_status,
                'attention_passed': 1 if attention_status == 'PASS' else 0,  # For optional filtering later
                'start_time': start_row['timestamp'],
                'response_time': response_row['timestamp'],
                'start_sample': start_row['sample']
            })
            
            events_list.append([start_row['sample'], 0, 1])
        
        metadata_df = pd.DataFrame(epoch_metadata)
        
        print(f"\nEpoch parameters:")
        print(f"  Time window: {tmin} to {tmax} s")
        print(f"  Baseline: {tmin} to 0 s")
        
        events = np.array(events_list)
        
        # Create epochs with ONLY artifact rejection (no attention filtering)
        epochs = mne.Epochs(
            self.raw,
            events,
            event_id={'statement': 1},
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0),
            preload=True,
            reject=dict(eeg=250e-6),  # Artifact rejection only
            verbose=False
        )
        
        # Report results
        n_passed_attn = metadata_df['attention_passed'].sum() if len(metadata_df) > 0 else 0
        n_failed_attn = len(metadata_df) - n_passed_attn
        
        print(f"\n✓ Created {len(epochs)} statement epochs")
        print(f"  Total before artifact rejection: {len(metadata_df)}")
        print(f"  Dropped (artifacts only): {len(metadata_df) - len(epochs)}")
        print(f"\n  Attention check metadata (NOT used for filtering):")
        print(f"    Passed: {n_passed_attn}")
        print(f"    Failed: {n_failed_attn}")
        
        if len(metadata_df) > 0:
            print(f"\n  Attention status breakdown:")
            attn_counts = metadata_df['attention_check'].value_counts()
            for status, count in attn_counts.items():
                print(f"    {status}: {count}")
        
        return epochs, metadata_df
    
    def create_article_epochs(self, tmin=-0.5, tmax=None) -> Tuple[mne.Epochs, pd.DataFrame]:
        """
        Create epochs for article reading.
        
        FILTERING POLICY: NONE (Keep all, metadata for later analysis)
        Rationale: Article SELECTION happens before attention checks.
        The confirmation bias decision is made at selection time.
        
        Attention check status preserved as metadata for optional filtering
        during reading-locked or post-rating analyses if needed.
        
        All epochs kept with attention_check status as metadata.
        """
        print(f"\n{'='*80}")
        print("STEP 4: CREATING ARTICLE EPOCHS (NO ATTENTION FILTERING)")
        print(f"{'='*80}")
        print("⚠️  All epochs kept: Selection happens BEFORE attention checks")
        
        # Find article markers
        topic_selects = self.markers_df[self.markers_df['category'] == 'TOPIC_SELECT']
        article_starts = self.markers_df[self.markers_df['category'] == 'ARTICLE_START']
        article_ratings = self.markers_df[self.markers_df['category'] == 'ARTICLE_RATING']
        
        print(f"Found {len(topic_selects)} topic selections")
        print(f"Found {len(article_starts)} article reads")
        print(f"Found {len(article_ratings)} article ratings")
        
        # Build metadata for ALL articles
        epoch_metadata = []
        events_list = []
        
        for idx, start_row in article_starts.iterrows():
            marker = start_row['marker']
            
            # Extract article code
            article_code = marker.split('_')[-1]
            topic_code = article_code[:3] if len(article_code) >= 3 else 'UNKNOWN'
            
            # Find corresponding rating
            next_ratings = article_ratings[
                (article_ratings['index'] > start_row['index']) &
                (article_ratings['marker'].str.contains(article_code))
            ]
            
            if len(next_ratings) == 0:
                continue
            
            rating_row = next_ratings.iloc[0]
            
            # Extract rating value
            rating_parts = [p for p in rating_row['marker'].split('_') 
                           if p.startswith('R') and len(p) <= 3 and len(p) > 1 and p[1].isdigit()]
            rating = int(rating_parts[0][1]) if rating_parts else 0
            
            # Extract reading time
            time_match = [p for p in rating_row['marker'].split('_') if 'TIME' in p]
            recorded_time = float(time_match[0].replace('TIME', '')) if time_match else 0
            
            # Find topic selection (closest before article start)
            prev_topics = topic_selects[topic_selects['index'] < start_row['index']]
            if len(prev_topics) > 0:
                topic_row = prev_topics.iloc[-1]
                epoch_start_sample = topic_row['sample']
                start_timestamp = topic_row['timestamp']
            else:
                epoch_start_sample = start_row['sample']
                start_timestamp = start_row['timestamp']
            
            # Look up attention check status (METADATA ONLY - NOT for filtering)
            attention_status = self.attention_check_map['articles'].get(article_code, 'MISSING')
            
            # Calculate duration
            duration = rating_row['timestamp'] - start_timestamp
            
            epoch_metadata.append({
                'article_code': article_code,
                'topic_code': topic_code,
                'rating': rating,
                'attention_check': attention_status,
                'attention_passed': 1 if attention_status == 'PASS' else 0,  # For optional filtering later
                'start_time': start_timestamp,
                'rating_time': rating_row['timestamp'],
                'duration': duration,
                'recorded_reading_time': recorded_time,
                'start_sample': epoch_start_sample
            })
            
            events_list.append([epoch_start_sample, 0, 1])
        
        metadata_df = pd.DataFrame(epoch_metadata)
        
        # Create epochs with fixed length
        if tmax is None:
            tmax = min(metadata_df['duration'].max(), 150.0) if len(metadata_df) > 0 else 150.0
        
        print(f"\nEpoch parameters:")
        print(f"  Baseline: {tmin} s")
        print(f"  Max duration: {tmax:.1f} s")
        if len(metadata_df) > 0:
            print(f"  Median actual duration: {metadata_df['duration'].median():.1f} s")
        
        events = np.array(events_list)
        
        # Create epochs with ONLY artifact rejection (no attention filtering)
        epochs = mne.Epochs(
            self.raw,
            events,
            event_id={'article': 1},
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0),
            preload=True,
            reject=dict(eeg=250e-6),  # Artifact rejection only
            verbose=False
        )
        
        # Report results
        n_passed_attn = metadata_df['attention_passed'].sum() if len(metadata_df) > 0 else 0
        n_failed_attn = len(metadata_df) - n_passed_attn
        
        print(f"\n✓ Created {len(epochs)} article epochs")
        print(f"  Total before artifact rejection: {len(metadata_df)}")
        print(f"  Dropped (artifacts only): {len(metadata_df) - len(epochs)}")
        print(f"\n  Attention check metadata (NOT used for filtering):")
        print(f"    Passed: {n_passed_attn}")
        print(f"    Failed: {n_failed_attn}")
        
        # Show breakdown
        if len(metadata_df) > 0:
            print(f"\n  Attention status breakdown:")
            attn_counts = metadata_df['attention_check'].value_counts()
            for status, count in attn_counts.items():
                print(f"    {status}: {count}")
        
        return epochs, metadata_df
    
    def save_epochs(self, epochs: mne.Epochs, metadata: pd.DataFrame, 
                   epoch_type: str):
        """Save epochs and metadata"""
        print(f"\n{'='*80}")
        print(f"SAVING {epoch_type.upper()} EPOCHS")
        print(f"{'='*80}")
        
        # Save epochs (MNE format)
        epochs_file = self.output_dir / f'{epoch_type}_epochs-epo.fif'
        epochs.save(epochs_file, overwrite=True, verbose=False)
        print(f"✓ Epochs saved: {epochs_file}")
        
        # Save metadata
        metadata_file = self.output_dir / f'{epoch_type}_metadata.csv'
        metadata.to_csv(metadata_file, index=False)
        print(f"✓ Metadata saved: {metadata_file}")
        
        # Save summary
        summary = {
            'total_epochs': len(epochs),
            'attention_passed': int(metadata['attention_passed'].sum()),
            'attention_failed': int(len(metadata) - metadata['attention_passed'].sum()),
            'mean_duration': float(metadata['duration'].mean()) if 'duration' in metadata.columns else None,
            'median_duration': float(metadata['duration'].median()) if 'duration' in metadata.columns else None,
            'max_duration': float(metadata['duration'].max()) if 'duration' in metadata.columns else None,
            'filtering_note': 'NO ATTENTION FILTERING - All epochs kept with metadata'
        }
        
        summary_file = self.output_dir / f'{epoch_type}_summary.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved: {summary_file}")
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        
        # Load data
        self.load_xdf()
        
        # Preprocess
        self.preprocess()
        
        # Create statement epochs (NO attention filtering)
        stmt_epochs, stmt_metadata = self.create_statement_epochs()
        self.save_epochs(stmt_epochs, stmt_metadata, 'statements')
        
        # Create article epochs (NO attention filtering)
        art_epochs, art_metadata = self.create_article_epochs()
        self.save_epochs(art_epochs, art_metadata, 'articles')
        
        print(f"\n{'='*80}")
        print("PREPROCESSING COMPLETE!")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nFiles created:")
        print(f"  statements_epochs-epo.fif")
        print(f"  statements_metadata.csv")
        print(f"  statements_summary.json")
        print(f"  articles_epochs-epo.fif")
        print(f"  articles_metadata.csv")
        print(f"  articles_summary.json")
        print(f"\n{'='*80}")
        print(f"CRITICAL REMINDER:")
        print(f"{'='*80}")
        print(f"✓ ALL epochs saved regardless of attention checks")
        print(f"✓ Use 'attention_passed' column in metadata for optional filtering")
        print(f"✓ Filtering decisions should be made during ANALYSIS based on:")
        print(f"  - Selection analysis: NO filtering (bias occurs before attention)")
        print(f"  - Reading analysis: OPTIONAL filtering (depends on hypothesis)")
        print(f"  - Post-rating analysis: CONSIDER filtering (engagement matters)")
        print(f"\nNext steps:")
        print(f"  1. Load epochs in analysis script")
        print(f"  2. Decide filtering strategy per analysis type")
        print(f"  3. Filter using: epochs[metadata['attention_passed'] == 1]")
        print(f"  4. Compute theta power, ERPs with appropriate filtering")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python eeg_preprocessing_pipeline.py <xdf_file> <output_dir>")
        sys.exit(1)
    
    xdf_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    preprocessor = EEGPreprocessor(xdf_file, output_dir)
    preprocessor.run_preprocessing()