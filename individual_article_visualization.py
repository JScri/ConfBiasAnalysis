#!/usr/bin/env python3
"""
INDIVIDUAL ARTICLE VISUALIZATION - 8 DECISION STAGES
===============================================================================
Creates comprehensive visualizations from individual article analysis:
1. ERP plots for each stage (all 14 articles overlaid)
2. Theta power heatmap (articles × stages)
3. Individual article trajectories across stages
4. Stage-specific bias comparisons

Reads output from individual_article_analysis.py

Author: Jason Stewart (25182902)
Ethics: ETH23-7909
Date: November 3, 2025
Version: 1.0
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color scheme for bias types
BIAS_COLORS = {
    'CONFIRMATION_BIAS': '#ef4444',     # Red
    'DISCONFIRMATION_SEEKING': '#3b82f6',  # Blue
    'COUNTER_BIAS': '#f59e0b',          # Orange
    'MIXED': '#6b7280',                 # Gray
    'NEUTRAL': '#10b981',               # Green
    'UNKNOWN': '#9ca3af'                # Light gray
}

# Stage order
STAGE_ORDER = [
    'topic_select',
    'selector_start',
    'article_preview',
    'article_select',
    'article_biosync',
    'reading_start',
    'reading_sustained',
    'article_rating'
]

STAGE_NAMES = {
    'topic_select': 'Topic Selection',
    'selector_start': 'List Presentation',
    'article_preview': 'Preview (Hover)',
    'article_select': 'Article Selection',
    'article_biosync': 'Reading Onset',
    'reading_start': 'Reading Start',
    'reading_sustained': 'Sustained Reading',
    'article_rating': 'Article Rating'
}


class IndividualArticleVisualizer:
    """
    Creates comprehensive visualizations for individual article analysis
    """
    
    def __init__(self, analysis_dir: str, output_dir: str = None):
        self.analysis_dir = Path(analysis_dir)
        
        if output_dir is None:
            self.output_dir = self.analysis_dir / 'visualizations'
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Data containers
        self.results = None
        self.theta_matrix = None
        self.erp_summary = None
        self.metadata = None
        
        print(f"\n{'='*80}")
        print(f"INDIVIDUAL ARTICLE VISUALIZATION")
        print(f"{'='*80}")
        print(f"Analysis directory: {self.analysis_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}\n")
    
    def load_analysis_results(self):
        """Load all analysis output files"""
        print(f"{'='*80}")
        print("LOADING ANALYSIS RESULTS")
        print(f"{'='*80}\n")
        
        # Load complete JSON results
        json_path = self.analysis_dir / 'individual_article_results.json'
        print(f"[1/4] Loading complete results...")
        with open(json_path, 'r') as f:
            self.results = json.load(f)
        print(f"  ✓ Loaded {len(self.results)} articles")
        
        # Load theta power matrix
        print(f"\n[2/4] Loading theta power matrix...")
        self.theta_matrix = pd.read_csv(self.analysis_dir / 'theta_power_matrix.csv')
        print(f"  ✓ Loaded theta matrix: {self.theta_matrix.shape}")
        
        # Load ERP summary
        print(f"\n[3/4] Loading ERP summary...")
        self.erp_summary = pd.read_csv(self.analysis_dir / 'erp_summary.csv')
        print(f"  ✓ Loaded {len(self.erp_summary)} ERP records")
        
        # Load metadata
        print(f"\n[4/4] Loading article metadata...")
        self.metadata = pd.read_csv(self.analysis_dir / 'article_metadata.csv')
        print(f"  ✓ Loaded {len(self.metadata)} articles")
        
        print(f"\n{'='*80}")
        print("DATA LOADING COMPLETE")
        print(f"{'='*80}\n")
    
    def plot_erps_by_stage(self):
        """
        Create 8 separate ERP plots, one per stage
        Each plot shows all 14 articles color-coded by bias type
        """
        print(f"{'='*80}")
        print("CREATING STAGE-SPECIFIC ERP PLOTS")
        print(f"{'='*80}\n")
        
        # Create 4x2 grid for 8 stages
        fig, axes = plt.subplots(4, 2, figsize=(18, 20))
        axes = axes.flatten()
        
        for idx, stage_id in enumerate(STAGE_ORDER):
            ax = axes[idx]
            stage_name = STAGE_NAMES[stage_id]
            
            print(f"  Plotting stage: {stage_name}")
            
            # Get all articles for this stage
            articles_plotted = 0
            
            for result in self.results:
                article_code = result['article_code']
                bias_type = result['bias_type']
                attention_pass = result['attention_pass']
                
                if stage_id in result['stages']:
                    stage_data = result['stages'][stage_id]
                    
                    if 'erp_frontal' in stage_data:
                        times = np.array(stage_data['erp_times'])
                        erp = np.array(stage_data['erp_frontal'])
                        
                        # Get color
                        color = BIAS_COLORS.get(bias_type, BIAS_COLORS['UNKNOWN'])
                        
                        # Line style based on attention check
                        linestyle = '-' if attention_pass else '--'
                        linewidth = 2.0 if attention_pass else 1.5
                        alpha = 0.8 if attention_pass else 0.5
                        
                        # Plot
                        bias_label = bias_type[:4] if (bias_type and isinstance(bias_type, str)) else 'N/A'
                        ax.plot(times, erp * 1e6,  # Convert to μV
                               color=color, linestyle=linestyle,
                               linewidth=linewidth, alpha=alpha,
                               label=f"{article_code} ({bias_label})")
                        
                        articles_plotted += 1
            
            # Format plot
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Time (s)', fontweight='bold', fontsize=10)
            ax.set_ylabel('Amplitude (μV)', fontweight='bold', fontsize=10)
            ax.set_title(f"{stage_name}\n(n={articles_plotted} articles)",
                        fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add phase label
            phase = result['stages'][stage_id].get('phase', 'unknown') if stage_id in result['stages'] else 'unknown'
            phase_color = '#3b82f6' if phase == 'selection' else '#10b981' if phase == 'reading' else '#ef4444'
            ax.text(0.02, 0.98, f"Phase: {phase.upper()}",
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=phase_color, alpha=0.3))
            
            print(f"    ✓ Plotted {articles_plotted} articles")
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=BIAS_COLORS['CONFIRMATION_BIAS'], lw=2, label='Confirmation Bias'),
            Line2D([0], [0], color=BIAS_COLORS['COUNTER_BIAS'], lw=2, label='Counter Bias'),
            Line2D([0], [0], color=BIAS_COLORS['MIXED'], lw=2, label='Mixed'),
            Line2D([0], [0], color=BIAS_COLORS['NEUTRAL'], lw=2, label='Neutral'),
            Line2D([0], [0], color='black', linestyle='-', lw=2, label='Attention PASS'),
            Line2D([0], [0], color='black', linestyle='--', lw=2, label='Attention FAIL')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=6, 
                  bbox_to_anchor=(0.5, -0.02), fontsize=11, frameon=True)
        
        plt.tight_layout(rect=[0, 0.02, 1, 1])
        
        output_path = self.output_dir / 'stage_specific_erps.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved: {output_path}")
    
    def plot_theta_heatmap(self):
        """
        Create heatmap showing theta power for all articles across all stages
        """
        print(f"\n{'='*80}")
        print("CREATING THETA POWER HEATMAP")
        print(f"{'='*80}\n")
        
        # Prepare data matrix
        theta_cols = [f'{stage}_theta' for stage in STAGE_ORDER]
        theta_data = self.theta_matrix[theta_cols].values
        
        # Convert to pV²/Hz for better visualization
        theta_data = theta_data * 1e12
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create heatmap
        im = ax.imshow(theta_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # Set ticks
        ax.set_xticks(np.arange(len(STAGE_ORDER)))
        ax.set_xticklabels([STAGE_NAMES[s] for s in STAGE_ORDER], rotation=45, ha='right')
        ax.set_yticks(np.arange(len(self.theta_matrix)))
        
        # Create y-tick labels with bias type and attention check
        y_labels = []
        for _, row in self.theta_matrix.iterrows():
            article = row['article_code']
            # Handle both None and pandas NaN values
            bias = row['bias_type'][:4] if pd.notna(row['bias_type']) and isinstance(row['bias_type'], str) else 'N/A'
            att = '✓' if row['attention_pass'] else '✗'
            y_labels.append(f"{article} ({bias}) {att}")
        ax.set_yticklabels(y_labels, fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Theta Power (pV²/Hz)', rotation=270, labelpad=20, fontweight='bold')
        
        # Add text annotations with theta values
        for i in range(len(self.theta_matrix)):
            for j in range(len(STAGE_ORDER)):
                value = theta_data[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value > theta_data[~np.isnan(theta_data)].mean() else 'black'
                    ax.text(j, i, f'{value:.2f}',
                           ha="center", va="center", color=text_color, fontsize=7)
        
        ax.set_xlabel('Stage', fontweight='bold', fontsize=12)
        ax.set_ylabel('Article', fontweight='bold', fontsize=12)
        ax.set_title('Theta Power Across Decision Stages\n(Frontal Channels, 4-8 Hz)',
                    fontweight='bold', fontsize=14, pad=20)
        
        # Add phase separators
        phase_boundaries = [3.5]  # Between selection and reading
        for boundary in phase_boundaries:
            ax.axvline(boundary, color='white', linewidth=3, linestyle='--')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'theta_power_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")
    
    def plot_article_trajectories(self):
        """
        Plot theta power trajectories for each article across stages
        """
        print(f"\n{'='*80}")
        print("CREATING ARTICLE TRAJECTORIES")
        print(f"{'='*80}\n")
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Get theta columns
        theta_cols = [f'{stage}_theta' for stage in STAGE_ORDER]
        x_positions = np.arange(len(STAGE_ORDER))
        
        # Plot each article
        for _, row in self.theta_matrix.iterrows():
            article_code = row['article_code']
            bias_type = row['bias_type']
            attention_pass = row['attention_pass']
            
            theta_values = row[theta_cols].values * 1e12  # Convert to pV²/Hz
            
            color = BIAS_COLORS.get(bias_type, BIAS_COLORS['UNKNOWN'])
            linestyle = '-' if attention_pass else '--'
            linewidth = 2.5 if attention_pass else 1.5
            alpha = 0.8 if attention_pass else 0.5
            marker = 'o' if attention_pass else 's'
            markersize = 8 if attention_pass else 6
            
            ax.plot(x_positions, theta_values,
                   color=color, linestyle=linestyle, linewidth=linewidth,
                   alpha=alpha, marker=marker, markersize=markersize,
                   label=f"{article_code} ({bias_type[:4] if (pd.notna(bias_type) and isinstance(bias_type, str)) else 'N/A'})")
        
        # Format
        ax.set_xticks(x_positions)
        ax.set_xticklabels([STAGE_NAMES[s] for s in STAGE_ORDER], rotation=45, ha='right')
        ax.set_xlabel('Stage', fontweight='bold', fontsize=12)
        ax.set_ylabel('Theta Power (pV²/Hz)', fontweight='bold', fontsize=12)
        ax.set_title('Individual Article Trajectories Across Decision Stages\n(Frontal Theta Power)',
                    fontweight='bold', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add phase regions
        ax.axvspan(-0.5, 3.5, alpha=0.1, color='blue', label='Selection Phase')
        ax.axvspan(3.5, 6.5, alpha=0.1, color='green', label='Reading Phase')
        ax.axvspan(6.5, 7.5, alpha=0.1, color='red', label='Response Phase')
        
        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'article_trajectories.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")
    
    def plot_bias_type_comparison(self):
        """
        Compare theta power across stages for different bias types
        """
        print(f"\n{'='*80}")
        print("CREATING BIAS TYPE COMPARISON")
        print(f"{'='*80}\n")
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        theta_cols = [f'{stage}_theta' for stage in STAGE_ORDER]
        
        for idx, (stage_id, theta_col) in enumerate(zip(STAGE_ORDER, theta_cols)):
            ax = axes[idx]
            stage_name = STAGE_NAMES[stage_id]
            
            # Group by bias type
            bias_groups = self.theta_matrix.groupby('bias_type')[theta_col].apply(list)
            
            # Prepare data for violin plot
            data_for_plot = []
            labels_for_plot = []
            colors_for_plot = []
            
            for bias_type, values in bias_groups.items():
                values_clean = [v * 1e12 for v in values if not np.isnan(v)]
                if len(values_clean) > 0:
                    data_for_plot.append(values_clean)
                    labels_for_plot.append(bias_type[:4] if (pd.notna(bias_type) and isinstance(bias_type, str)) else 'N/A')  # Abbreviate
                    colors_for_plot.append(BIAS_COLORS.get(bias_type, BIAS_COLORS['UNKNOWN']))
            
            # Create violin plot
            if len(data_for_plot) > 0:
                parts = ax.violinplot(data_for_plot, positions=range(len(data_for_plot)),
                                     showmeans=True, showmedians=True)
                
                # Color the violins
                for pc, color in zip(parts['bodies'], colors_for_plot):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax.set_xticks(range(len(labels_for_plot)))
                ax.set_xticklabels(labels_for_plot, rotation=45, ha='right')
                ax.set_ylabel('Theta Power (pV²/Hz)', fontsize=9)
                ax.set_title(stage_name, fontweight='bold', fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Theta Power by Bias Type Across All Stages',
                    fontweight='bold', fontsize=16, y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / 'bias_type_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")
    
    def plot_attention_check_comparison(self):
        """
        Compare articles that passed vs failed attention checks
        """
        print(f"\n{'='*80}")
        print("CREATING ATTENTION CHECK COMPARISON")
        print(f"{'='*80}\n")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        theta_cols = [f'{stage}_theta' for stage in STAGE_ORDER]
        x_positions = np.arange(len(STAGE_ORDER))
        
        # Separate by attention check
        pass_data = self.theta_matrix[self.theta_matrix['attention_pass'] == True]
        fail_data = self.theta_matrix[self.theta_matrix['attention_pass'] == False]
        
        # Calculate means and SEMs
        pass_means = pass_data[theta_cols].mean() * 1e12
        pass_sems = pass_data[theta_cols].sem() * 1e12
        fail_means = fail_data[theta_cols].mean() * 1e12
        fail_sems = fail_data[theta_cols].sem() * 1e12
        
        # Plot
        width = 0.35
        ax.bar(x_positions - width/2, pass_means, width,
              yerr=pass_sems, capsize=5, label=f'Attention PASS (n={len(pass_data)})',
              color='#10b981', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.bar(x_positions + width/2, fail_means, width,
              yerr=fail_sems, capsize=5, label=f'Attention FAIL (n={len(fail_data)})',
              color='#ef4444', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels([STAGE_NAMES[s] for s in STAGE_ORDER], rotation=45, ha='right')
        ax.set_xlabel('Stage', fontweight='bold', fontsize=12)
        ax.set_ylabel('Mean Theta Power (pV²/Hz)', fontweight='bold', fontsize=12)
        ax.set_title('Theta Power: Attention Check Comparison Across Stages',
                    fontweight='bold', fontsize=14, pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add phase separators
        ax.axvline(3.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.text(1.5, ax.get_ylim()[1] * 0.95, 'Selection\nPhase',
               ha='center', fontsize=10, fontweight='bold')
        ax.text(5, ax.get_ylim()[1] * 0.95, 'Reading\nPhase',
               ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'attention_check_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")
    
    def create_summary_report(self):
        """Generate text summary report"""
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'='*80}\n")
        
        report_path = self.output_dir / 'VISUALIZATION_SUMMARY.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("INDIVIDUAL ARTICLE VISUALIZATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write("-"*80 + "\n")
            f.write(f"Total articles analyzed: {len(self.metadata)}\n")
            f.write(f"Attention checks PASS: {self.metadata['attention_pass'].sum()}\n")
            f.write(f"Attention checks FAIL: {(~self.metadata['attention_pass']).sum()}\n\n")
            
            f.write("BIAS TYPE DISTRIBUTION:\n")
            f.write("-"*80 + "\n")
            bias_counts = self.metadata['bias_type'].value_counts()
            for bias_type, count in bias_counts.items():
                f.write(f"  {bias_type}: {count}\n")
            f.write("\n")
            
            f.write("STAGES ANALYZED:\n")
            f.write("-"*80 + "\n")
            for idx, stage_id in enumerate(STAGE_ORDER, 1):
                f.write(f"  {idx}. {STAGE_NAMES[stage_id]}\n")
            f.write("\n")
            
            f.write("VISUALIZATIONS GENERATED:\n")
            f.write("-"*80 + "\n")
            f.write("  1. stage_specific_erps.png - ERP waveforms for each stage\n")
            f.write("  2. theta_power_heatmap.png - Theta power matrix\n")
            f.write("  3. article_trajectories.png - Individual trajectories\n")
            f.write("  4. bias_type_comparison.png - Bias type comparisons\n")
            f.write("  5. attention_check_comparison.png - Attention effect\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
        
        print(f"✓ Saved: {report_path}")
    
    def create_all_visualizations(self):
        """Generate all visualizations"""
        print(f"\n{'='*80}")
        print("CREATING ALL VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        self.load_analysis_results()
        self.plot_erps_by_stage()
        self.plot_theta_heatmap()
        self.plot_article_trajectories()
        self.plot_bias_type_comparison()
        self.plot_attention_check_comparison()
        self.create_summary_report()
        
        print(f"\n{'='*80}")
        print("ALL VISUALIZATIONS COMPLETE")
        print(f"{'='*80}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize Individual Article Analysis Results'
    )
    
    parser.add_argument('--analysis-dir', required=True,
                       help='Directory containing analysis results')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory (default: analysis_dir/visualizations)')
    
    args = parser.parse_args()
    
    visualizer = IndividualArticleVisualizer(
        analysis_dir=args.analysis_dir,
        output_dir=args.output
    )
    
    visualizer.create_all_visualizations()