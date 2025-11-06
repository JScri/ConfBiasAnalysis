#!/usr/bin/env python3
"""
Comprehensive Event Marker Extraction and Visualization
BCI Confirmation Bias Study - UTS 2025

This script extracts ALL event markers from XDF files and provides:
- Complete marker timeline with timestamps
- Phase-by-phase breakdown
- Marker statistics and counts
- Visual timeline visualization
- CSV export for further analysis

Author: Jason Stewart (25182902)
Ethics: ETH23-7909
"""

import pyxdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import json
import sys
import os
from collections import Counter, defaultdict

class XDFMarkerExtractor:
    """Comprehensive marker extraction and analysis"""
    
    def __init__(self, xdf_file):
        """
        Initialize extractor with XDF file
        
        Args:
            xdf_file: Path to XDF file
        """
        self.xdf_file = xdf_file
        self.markers_df = None
        self.phase_breakdown = {}
        self.marker_stats = {}
        
        # Define marker categories for classification
        self.marker_categories = {
            'EXPERIMENT': ['EXPERIMENT_START', 'EXPERIMENT_END', 'LSL_STREAMS'],
            'APPLICATION': ['APPLICATION_RESUMED', 'APPLICATION_GAINED_FOCUS', 'APPLICATION_LOST_FOCUS'],
            'PHASE1': ['PHASE1_START', 'PHASE1_PRIMING_START', 'PHASE1_END', 'PHASE1_PRIMING_END'],
            'PHASE2': ['PHASE2_START', 'PHASE2_EVIDENCE_START', 'PHASE2_END', 'PHASE2_EVIDENCE_END'],
            'STATEMENT': ['STATEMENT_PRESENT', 'STATEMENT_RESPONSE', 'FIXATION_ONSET'],
            'ATTENTION': ['ATTENTION_CHECK'],
            'REST': ['REST_BREAK_START', 'REST_BREAK_END'],
            'ARTICLE': ['ARTICLE_READ_START', 'ARTICLE_READ_END', 'ARTICLE_SELECT', 
                       'ARTICLE_PREVIEW', 'ARTICLE_RATING', 'ARTICLE_SCROLL'],
            'TOPIC': ['TOPIC_SELECT', 'DROPDOWN_OPEN'],
            'INSTRUCTION': ['INSTRUCTION_CONTINUE', 'INSTRUCTION_SCREEN'],
            'BIAS': ['BIAS_EXPECTED', 'BIAS_ACTUAL', 'ACTUAL'],
            'BIOSYNC': ['BIOSYNC'],  # Biosignal synchronization markers
            'FINAL': ['FINAL_RESPONSE']
        }
        
    def load_xdf(self):
        """Load XDF file and extract marker stream"""
        print(f"\n{'='*80}")
        print(f"LOADING XDF FILE")
        print(f"{'='*80}")
        print(f"File: {self.xdf_file}")
        print(f"Size: {os.path.getsize(self.xdf_file) / 1024 / 1024:.2f} MB\n")
        
        try:
            streams, header = pyxdf.load_xdf(self.xdf_file)
            print(f"✓ Successfully loaded {len(streams)} streams")
        except Exception as e:
            print(f"✗ Error loading XDF: {e}")
            return False
        
        # Find marker stream
        marker_stream = None
        for stream in streams:
            stream_name = stream['info']['name'][0] if 'name' in stream['info'] else 'Unknown'
            if 'marker' in stream_name.lower():
                marker_stream = stream
                print(f"✓ Found marker stream: {stream_name}")
                break
        
        if marker_stream is None:
            print("✗ No marker stream found in XDF file!")
            return False
        
        # Extract markers
        self._extract_markers(marker_stream)
        return True
    
    def _extract_markers(self, stream):
        """Extract markers from stream into DataFrame"""
        time_stamps = stream['time_stamps']
        time_series = stream['time_series']
        
        markers_list = []
        for idx, (timestamp, marker_data) in enumerate(zip(time_stamps, time_series)):
            # Extract marker string
            marker_str = marker_data[0] if isinstance(marker_data, (list, np.ndarray)) else str(marker_data)
            
            # Categorize marker
            category = self._categorize_marker(marker_str)
            
            # Parse marker details
            details = self._parse_marker_details(marker_str)
            
            markers_list.append({
                'index': idx,
                'timestamp': timestamp,
                'marker': marker_str,
                'category': category,
                **details
            })
        
        self.markers_df = pd.DataFrame(markers_list)
        
        # Add relative time (from first marker)
        self.markers_df['relative_time'] = self.markers_df['timestamp'] - self.markers_df['timestamp'].iloc[0]
        
        print(f"✓ Extracted {len(self.markers_df)} markers")
        print(f"  Duration: {self.markers_df['relative_time'].iloc[-1]:.1f}s")
        print(f"  First marker: {self.markers_df['marker'].iloc[0]}")
        print(f"  Last marker: {self.markers_df['marker'].iloc[-1]}")
    
    def _categorize_marker(self, marker_str):
        """Categorize marker based on its content"""
        for category, patterns in self.marker_categories.items():
            for pattern in patterns:
                if pattern in marker_str:
                    return category
        return 'OTHER'
    
    def _parse_marker_details(self, marker_str):
        """Parse marker string to extract detailed information"""
        details = {
            'topic_code': None,
            'statement_id': None,
            'article_code': None,
            'rating': None,
            'bias_type': None,
            'response_time': None
        }
        
        # Extract topic code (e.g., T01, T02, T15, T20)
        if '_T' in marker_str:
            parts = marker_str.split('_')
            for part in parts:
                # Topic code is T followed by 1-2 digits
                if part.startswith('T') and len(part) >= 2:
                    # Extract just the topic portion (T + digits)
                    topic_part = part[1:]  # Remove 'T'
                    digits = ''
                    for char in topic_part:
                        if char.isdigit():
                            digits += char
                        else:
                            break
                    if digits:
                        details['topic_code'] = 'T' + digits
                        break
        
        # Extract statement ID (e.g., Q0_T04_S01 -> S01)
        if '_S' in marker_str:
            parts = marker_str.split('_S')
            if len(parts) > 1:
                statement_part = parts[1].split('_')[0]
                details['statement_id'] = 'S' + statement_part
        
        # Extract article code (e.g., T01A, T02B)
        if 'ARTICLE' in marker_str:
            parts = marker_str.split('_')
            for part in parts:
                if part.startswith('T') and len(part) >= 4 and part[3].isalpha():
                    details['article_code'] = part[:4]
                    break
        
        # Extract rating (e.g., AGREEMENT_5, RATING_3)
        if 'AGREEMENT_' in marker_str or 'RATING_' in marker_str:
            parts = marker_str.split('_')
            for part in parts:
                if part.isdigit() and int(part) <= 5:
                    details['rating'] = int(part)
                    break
        
        # Extract bias type
        if 'CONFIRMATORY' in marker_str:
            details['bias_type'] = 'confirmatory'
        elif 'DISCONFIRMATORY' in marker_str:
            details['bias_type'] = 'disconfirmatory'
        elif 'NEUTRAL' in marker_str:
            details['bias_type'] = 'neutral'
        
        return details
    
    def analyze_phases(self):
        """Analyze markers by experimental phase"""
        print(f"\n{'='*80}")
        print(f"PHASE ANALYSIS")
        print(f"{'='*80}\n")
        
        # Find phase boundaries
        phase1_start = self.markers_df[self.markers_df['marker'].str.contains('PHASE1.*START', case=False, na=False)]
        phase1_end = self.markers_df[self.markers_df['marker'].str.contains('PHASE1.*END', case=False, na=False)]
        phase2_start = self.markers_df[self.markers_df['marker'].str.contains('PHASE2.*START', case=False, na=False)]
        phase2_end = self.markers_df[self.markers_df['marker'].str.contains('PHASE2.*END', case=False, na=False)]
        
        # Phase 1 analysis
        if not phase1_start.empty and not phase1_end.empty:
            p1_start_time = phase1_start.iloc[0]['timestamp']
            p1_end_time = phase1_end.iloc[0]['timestamp']
            p1_markers = self.markers_df[
                (self.markers_df['timestamp'] >= p1_start_time) & 
                (self.markers_df['timestamp'] <= p1_end_time)
            ]
            
            print(f"📋 PHASE 1: Statement Presentation (Priming)")
            print(f"  Duration: {p1_end_time - p1_start_time:.1f}s")
            print(f"  Total markers: {len(p1_markers)}")
            print(f"  Start: {p1_start_time:.3f}s")
            print(f"  End: {p1_end_time:.3f}s\n")
            
            # Count statement presentations and responses
            statements = p1_markers[p1_markers['category'] == 'STATEMENT']
            presentations = statements[statements['marker'].str.contains('PRESENT', na=False)]
            responses = statements[statements['marker'].str.contains('RESPONSE', na=False)]
            attention_checks = p1_markers[p1_markers['category'] == 'ATTENTION']
            rest_breaks = p1_markers[p1_markers['category'] == 'REST']
            
            print(f"  📊 Statement presentations: {len(presentations)}")
            print(f"  ✍️  Statement responses: {len(responses)}")
            print(f"  ⚠️  Attention checks: {len(attention_checks)}")
            print(f"  ☕ Rest breaks: {len(rest_breaks) // 2}")
            
            self.phase_breakdown['phase1'] = {
                'duration': p1_end_time - p1_start_time,
                'markers': p1_markers,
                'presentations': presentations,
                'responses': responses,
                'attention_checks': attention_checks,
                'rest_breaks': rest_breaks
            }
        
        # Phase 2 analysis
        if not phase2_start.empty and not phase2_end.empty:
            p2_start_time = phase2_start.iloc[0]['timestamp']
            p2_end_time = phase2_end.iloc[0]['timestamp']
            p2_markers = self.markers_df[
                (self.markers_df['timestamp'] >= p2_start_time) & 
                (self.markers_df['timestamp'] <= p2_end_time)
            ]
            
            print(f"\n📰 PHASE 2: Evidence Presentation (Article Reading)")
            print(f"  Duration: {p2_end_time - p2_start_time:.1f}s")
            print(f"  Total markers: {len(p2_markers)}")
            print(f"  Start: {p2_start_time:.3f}s")
            print(f"  End: {p2_end_time:.3f}s\n")
            
            # Count article events
            article_markers = p2_markers[p2_markers['category'] == 'ARTICLE']
            read_starts = article_markers[article_markers['marker'].str.contains('READ_START', na=False)]
            read_ends = article_markers[article_markers['marker'].str.contains('READ_END', na=False)]
            selections = article_markers[article_markers['marker'].str.contains('SELECT', na=False)]
            ratings = article_markers[article_markers['marker'].str.contains('RATING', na=False)]
            
            print(f"  📖 Articles read (starts): {len(read_starts)}")
            print(f"  ✅ Articles completed (ends): {len(read_ends)}")
            print(f"  🎯 Article selections: {len(selections)}")
            print(f"  ⭐ Article ratings: {len(ratings)}")
            
            self.phase_breakdown['phase2'] = {
                'duration': p2_end_time - p2_start_time,
                'markers': p2_markers,
                'read_starts': read_starts,
                'read_ends': read_ends,
                'selections': selections,
                'ratings': ratings
            }
    
    def generate_statistics(self):
        """Generate comprehensive marker statistics"""
        print(f"\n{'='*80}")
        print(f"MARKER STATISTICS")
        print(f"{'='*80}\n")
        
        # Category counts
        print("📊 Markers by Category:")
        category_counts = self.markers_df['category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category:15s}: {count:4d}")
        
        # Most frequent markers
        print(f"\n🔝 Top 10 Most Frequent Markers:")
        marker_counts = Counter(self.markers_df['marker'])
        for marker, count in marker_counts.most_common(10):
            # Truncate long markers
            display_marker = marker[:60] + '...' if len(marker) > 60 else marker
            print(f"  {count:4d}x  {display_marker}")
        
        # Temporal statistics
        print(f"\n⏱️  Temporal Statistics:")
        inter_marker_intervals = np.diff(self.markers_df['timestamp'])
        print(f"  Mean interval: {np.mean(inter_marker_intervals):.3f}s")
        print(f"  Median interval: {np.median(inter_marker_intervals):.3f}s")
        print(f"  Min interval: {np.min(inter_marker_intervals):.3f}s")
        print(f"  Max interval: {np.max(inter_marker_intervals):.3f}s")
        
        # Topic distribution
        topics = self.markers_df[self.markers_df['topic_code'].notna()]['topic_code']
        if not topics.empty:
            print(f"\n🗂️  Topics Covered:")
            topic_counts = topics.value_counts().sort_index()
            for topic, count in topic_counts.items():
                print(f"  {topic}: {count} markers")
        
        # Rating distribution
        ratings = self.markers_df[self.markers_df['rating'].notna()]['rating']
        if not ratings.empty:
            print(f"\n⭐ Rating Distribution:")
            rating_counts = ratings.value_counts().sort_index()
            for rating, count in rating_counts.items():
                print(f"  Rating {int(rating)}: {count} responses")
        
        self.marker_stats = {
            'category_counts': category_counts.to_dict(),
            'top_markers': dict(marker_counts.most_common(10)),
            'temporal': {
                'mean_interval': float(np.mean(inter_marker_intervals)),
                'median_interval': float(np.median(inter_marker_intervals)),
                'min_interval': float(np.min(inter_marker_intervals)),
                'max_interval': float(np.max(inter_marker_intervals))
            }
        }
    
    def visualize_timeline(self, output_dir=None):
        """Create visual timeline of markers"""
        print(f"\n{'='*80}")
        print(f"CREATING TIMELINE VISUALIZATION")
        print(f"{'='*80}\n")
        
        if output_dir is None:
            output_dir = os.path.dirname(self.xdf_file)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Define colors for categories
        category_colors = {
            'EXPERIMENT': '#1f77b4',
            'PHASE1': '#ff7f0e',
            'PHASE2': '#2ca02c',
            'STATEMENT': '#d62728',
            'ARTICLE': '#9467bd',
            'ATTENTION': '#8c564b',
            'REST': '#e377c2',
            'TOPIC': '#7f7f7f',
            'APPLICATION': '#bcbd22',
            'INSTRUCTION': '#17becf',
            'BIAS': '#ff9896',
            'OTHER': '#c5b0d5'
        }
        
        # Plot 1: Marker density over time
        times = self.markers_df['relative_time'].values
        categories = self.markers_df['category'].values
        
        for category in self.markers_df['category'].unique():
            cat_times = times[categories == category]
            ax1.scatter(cat_times, [category] * len(cat_times), 
                       c=category_colors.get(category, '#cccccc'),
                       s=50, alpha=0.6, label=category)
        
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Marker Category', fontsize=12)
        ax1.set_title('Event Marker Timeline by Category', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative marker count by category
        for category in sorted(self.markers_df['category'].unique()):
            cat_df = self.markers_df[self.markers_df['category'] == category]
            cumulative_count = range(1, len(cat_df) + 1)
            ax2.plot(cat_df['relative_time'], cumulative_count,
                    label=category, linewidth=2, alpha=0.7,
                    color=category_colors.get(category, '#cccccc'))
        
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Cumulative Count', fontsize=12)
        ax2.set_title('Cumulative Marker Count Over Time', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, 'marker_timeline.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved timeline to: {output_file}")
        plt.close()
        
        # Create phase-specific timeline
        self._create_phase_timeline(output_dir)
    
    def _create_phase_timeline(self, output_dir):
        """Create detailed phase-by-phase timeline"""
        if not self.phase_breakdown:
            return
        
        fig, axes = plt.subplots(len(self.phase_breakdown), 1, 
                                figsize=(14, 5 * len(self.phase_breakdown)))
        
        if len(self.phase_breakdown) == 1:
            axes = [axes]
        
        for idx, (phase_name, phase_data) in enumerate(self.phase_breakdown.items()):
            ax = axes[idx]
            phase_markers = phase_data['markers']
            
            # Plot markers
            rel_time = phase_markers['relative_time'].values
            categories = phase_markers['category'].values
            
            unique_cats = phase_markers['category'].unique()
            y_positions = {cat: i for i, cat in enumerate(unique_cats)}
            
            for category in unique_cats:
                cat_times = rel_time[categories == category]
                y_vals = [y_positions[category]] * len(cat_times)
                ax.scatter(cat_times, y_vals, s=50, alpha=0.6, label=category)
            
            ax.set_xlabel('Time (seconds)', fontsize=11)
            ax.set_ylabel('Event Type', fontsize=11)
            ax.set_title(f'{phase_name.upper()}: Event Timeline', 
                        fontsize=12, fontweight='bold')
            ax.set_yticks(list(y_positions.values()))
            ax.set_yticklabels(list(y_positions.keys()))
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'phase_timelines.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved phase timelines to: {output_file}")
        plt.close()
    
    def export_to_csv(self, output_dir=None):
        """Export markers to CSV files"""
        print(f"\n{'='*80}")
        print(f"EXPORTING TO CSV")
        print(f"{'='*80}\n")
        
        if output_dir is None:
            output_dir = os.path.dirname(self.xdf_file)
        
        # Export all markers
        all_markers_file = os.path.join(output_dir, 'all_markers.csv')
        self.markers_df.to_csv(all_markers_file, index=False)
        print(f"✓ Exported all markers to: {all_markers_file}")
        
        # Export by category
        for category in self.markers_df['category'].unique():
            cat_markers = self.markers_df[self.markers_df['category'] == category]
            cat_file = os.path.join(output_dir, f'markers_{category.lower()}.csv')
            cat_markers.to_csv(cat_file, index=False)
            print(f"  - {category}: {cat_file}")
        
        # Export phase-specific markers
        for phase_name, phase_data in self.phase_breakdown.items():
            phase_file = os.path.join(output_dir, f'markers_{phase_name}.csv')
            phase_data['markers'].to_csv(phase_file, index=False)
            print(f"  - {phase_name.upper()}: {phase_file}")
        
        # Export summary statistics
        stats_file = os.path.join(output_dir, 'marker_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(self.marker_stats, f, indent=2)
        print(f"✓ Exported statistics to: {stats_file}")
    
    def print_marker_list(self, max_markers=50):
        """Print formatted list of markers"""
        print(f"\n{'='*80}")
        print(f"COMPLETE MARKER LIST (showing first {max_markers})")
        print(f"{'='*80}\n")
        
        for idx, row in self.markers_df.head(max_markers).iterrows():
            category_tag = f"[{row['category']:10s}]"
            time_tag = f"{row['relative_time']:8.2f}s"
            marker_text = row['marker'][:60]
            
            print(f"{idx:4d}  {time_tag}  {category_tag}  {marker_text}")
        
        if len(self.markers_df) > max_markers:
            print(f"\n... and {len(self.markers_df) - max_markers} more markers")
    
    def run_complete_analysis(self):
        """Run all analysis steps"""
        if not self.load_xdf():
            return False
        
        self.analyze_phases()
        self.generate_statistics()
        self.print_marker_list()
        self.visualize_timeline()
        self.export_to_csv()
        
        print(f"\n{'='*80}")
        print(f"✅ ANALYSIS COMPLETE!")
        print(f"{'='*80}\n")
        return True


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("\nXDF Marker Extraction and Visualization Tool")
        print("=" * 50)
        print("\nUsage:")
        print("  python xdf_marker_extractor.py <xdf_file>")
        print("\nExample:")
        print("  python xdf_marker_extractor.py test_session11.xdf")
        print("\nOutputs:")
        print("  - all_markers.csv: Complete marker list with timestamps")
        print("  - markers_<category>.csv: Markers by category")
        print("  - markers_phase1/phase2.csv: Phase-specific markers")
        print("  - marker_timeline.png: Visual timeline")
        print("  - phase_timelines.png: Phase-by-phase timeline")
        print("  - marker_statistics.json: Summary statistics")
        sys.exit(1)
    
    xdf_file = sys.argv[1]
    
    if not os.path.exists(xdf_file):
        print(f"\n✗ Error: File not found: {xdf_file}")
        sys.exit(1)
    
    # Run analysis
    extractor = XDFMarkerExtractor(xdf_file)
    success = extractor.run_complete_analysis()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()