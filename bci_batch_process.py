#!/usr/bin/env python3
"""
BCI Confirmation Bias - Batch Processing Pipeline
Updated for new directory structure: test_session*/ folders in current directory

VERSION: 3.0 - CORRECTED METHODOLOGY (November 2, 2025)
Changes from v2.0:
- Added new metrics from corrected behavioral analysis
- n_total_articles: denominator for CB% calculation
- classifiable_rate: proportion of articles with valid statement linkages
- Attention checks tracked separately for quality/EEG purposes

Author: Jason Stewart
Ethics: ETH23-7909
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import analysis functions from existing scripts
from bci_analysis_behavioural import (
    load_json_data, calculate_bias_metrics,
    XDFReconstructor, create_comprehensive_visualizations,
    StatementResponse, ArticleResponse
)

class ParticipantSession:
    """Represents a single participant session with all data"""
    
    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir)
        self.session_name = self.session_dir.name
        self.participant_id = self._extract_participant_id()
        self.has_eeg = 'eeg_' in self.session_name.lower()
        
        # File paths
        self.json_file = None
        self.xdf_file = None
        
        # Data containers
        self.statements = []
        self.articles = []
        self.bias_metrics = {}
        
        # Find files
        self._locate_files()
    
    def _extract_participant_id(self) -> str:
        """Extract participant ID from session name"""
        # Extract number from session name (e.g., "11" from "test_session11")
        import re
        match = re.search(r'(\d+)$', self.session_name)
        if match:
            return match.group(1)
        return self.session_name
    
    def _locate_files(self):
        """Locate all relevant files in session directory"""
        if not self.session_dir.exists():
            print(f"  ⚠ Warning: Directory {self.session_dir} does not exist")
            return
        
        # Find participant JSON file with flexible naming
        # Looking for participant_11.json, participant_12.json, etc.
        json_patterns = [
            f'participant_{self.participant_id}.json',  # participant_11.json
            f'participant_*{self.participant_id}*.json',  # any participant file with ID
            'participant_*.json',  # fallback to any participant file
            '*.json'  # last resort: any JSON file
        ]
        
        for pattern in json_patterns:
            json_files = list(self.session_dir.glob(pattern))
            if json_files:
                self.json_file = json_files[0]
                print(f"  ✓ Found JSON: {self.json_file.name}")
                break
        
        if not self.json_file:
            print(f"  ✗ No JSON file found in {self.session_dir}")
            print(f"     Searched for patterns: {json_patterns}")
        
        # Find XDF file
        xdf_patterns = [
            f'{self.session_name}.xdf',  # test_session11.xdf
            '*.xdf'  # any XDF file
        ]
        
        for pattern in xdf_patterns:
            xdf_files = list(self.session_dir.glob(pattern))
            if xdf_files:
                self.xdf_file = xdf_files[0]
                print(f"  ✓ Found XDF: {self.xdf_file.name}")
                break
        
        if not self.xdf_file:
            print(f"  ⚠ No XDF file found in {self.session_dir}")
    
    def load_behavioral_data(self) -> bool:
        """Load behavioral data from available sources"""
        if not self.json_file:
            print(f"  ✗ No JSON file found in {self.session_dir}")
            return False
        
        try:
            print(f"  Loading behavioral data from {self.json_file.name}")
            self.statements, self.articles = load_json_data(str(self.json_file))
            
            # If XDF exists and has more data, use it for timing
            if self.xdf_file and self.xdf_file.exists():
                print(f"  Attempting to load timing from {self.xdf_file.name}")
                try:
                    reconstructor = XDFReconstructor(str(self.xdf_file))
                    xdf_statements, xdf_articles = reconstructor.reconstruct_responses()
                    
                    # Only override if XDF has data
                    if xdf_statements:
                        print(f"    ✓ Using XDF timing for {len(xdf_statements)} statements")
                        self.statements = xdf_statements
                    if xdf_articles:
                        print(f"    ✓ Using XDF timing for {len(xdf_articles)} articles")
                        self.articles = xdf_articles
                except Exception as e:
                    print(f"    ⚠ Warning: Could not load XDF: {str(e)[:100]}")
            
            print(f"  ✓ Loaded {len(self.statements)} statements, {len(self.articles)} articles")
            return True
            
        except Exception as e:
            print(f"  ✗ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_metrics(self):
        """Calculate bias metrics"""
        if self.statements and self.articles:
            self.bias_metrics = calculate_bias_metrics(self.statements, self.articles)
            print(f"  ✓ Calculated bias metrics: CB rate = {self.bias_metrics['confirmation_bias_rate']:.1%}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics for this participant"""
        summary = {
            'participant_id': self.participant_id,
            'session_name': self.session_name,
            'has_eeg': self.has_eeg,
            'n_statements': len(self.statements),
            'n_articles': len(self.articles),
        }
        
        if self.bias_metrics:
            summary.update({
                # Primary metrics (corrected methodology v4.0)
                'confirmation_bias_rate': self.bias_metrics['confirmation_bias_rate'],
                'disconfirmation_rate': self.bias_metrics['disconfirmation_seeking_rate'],
                
                # Quality metrics
                'attention_pass_rate': self.bias_metrics['attention_pass_rate'],
                'classifiable_rate': self.bias_metrics.get('classifiable_rate', 0),
                
                # Counts
                'n_total_articles': self.bias_metrics.get('n_total_articles', 0),
                'n_classifiable': self.bias_metrics.get('n_classifiable', 0),
                'n_valid_pairs': self.bias_metrics['n_valid_pairs']
            })
        
        # Add RT statistics
        if self.statements:
            valid_stmt_rts = [s.reaction_time for s in self.statements 
                             if s.attention_check == 'PASS']
            if valid_stmt_rts:
                summary['mean_statement_rt'] = np.mean(valid_stmt_rts)
                summary['std_statement_rt'] = np.std(valid_stmt_rts)
        
        if self.articles:
            valid_art_rts = [a.reading_time for a in self.articles 
                            if a.attention_check == 'PASS']
            if valid_art_rts:
                summary['mean_article_rt'] = np.mean(valid_art_rts)
                summary['std_article_rt'] = np.std(valid_art_rts)
        
        return summary

class BCIBatchProcessor:
    """Process multiple participant sessions in batch"""
    
    def __init__(self, base_dir: str = './', output_dir: str = './bci_output/patched/batch_behavioural/'):
        """
        Initialize batch processor
        
        Args:
            base_dir: Directory containing test_session folders (default: current directory)
            output_dir: Where to save results (default: ./bci_output/patched/batch_behavioural/)
        """
        self.base_dir = Path(base_dir).absolute()
        self.output_dir = Path(output_dir).absolute()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.sessions = []
        self.summary_df = None
        
        print(f"\n{'='*70}")
        print("BCI BATCH PROCESSOR - CONFIGURATION")
        print('='*70)
        print(f"Base directory:   {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def validate_directory_structure(self):
        """Validate that the directory structure is as expected"""
        print("Validating directory structure...")
        
        # Check if base directory exists
        if not self.base_dir.exists():
            print(f"  ✗ Base directory does not exist: {self.base_dir}")
            return False
        
        # Look for test_session directories
        test_sessions = [d for d in self.base_dir.glob('test_session*') if d.is_dir()]
        
        # Count how many items were filtered out
        all_matches = list(self.base_dir.glob('test_session*'))
        filtered_out = len(all_matches) - len(test_sessions)
        
        if filtered_out > 0:
            print(f"  ℹ Filtered out {filtered_out} non-directory items (e.g., .xdf files in current directory)")
        
        if not test_sessions:
            print(f"  ⚠ No test_session* directories found in {self.base_dir}")
            print(f"\n  Expected structure:")
            print(f"    {self.base_dir}/")
            print(f"    ├── test_session11/")
            print(f"    │   ├── test_session11.xdf")
            print(f"    │   └── participant_11.json")
            print(f"    ├── test_session12/")
            print(f"    │   ├── test_session12.xdf")
            print(f"    │   └── participant_12.json")
            print(f"    └── ...")
            print(f"\n  Available directories:")
            for item in self.base_dir.iterdir():
                if item.is_dir():
                    print(f"    - {item.name}/")
            return False
        
        print(f"  ✓ Found {len(test_sessions)} test_session directories")
        
        # Check each session
        for session_dir in sorted(test_sessions):
            print(f"\n  Checking {session_dir.name}/")
            
            # Check for XDF
            xdf_files = list(session_dir.glob('*.xdf'))
            if xdf_files:
                print(f"    ✓ XDF: {xdf_files[0].name}")
            else:
                print(f"    ✗ No XDF file found")
            
            # Check for JSON
            json_files = list(session_dir.glob('*.json'))
            if json_files:
                print(f"    ✓ JSON: {json_files[0].name}")
            else:
                print(f"    ✗ No JSON file found")
        
        return True
    
    def find_sessions(self, patterns: List[str] = None):
        """Find all session directories matching patterns"""
        if patterns is None:
            patterns = ['test_session*', 'eeg_session*']
        
        session_dirs = []
        for pattern in patterns:
            found = list(self.base_dir.glob(pattern))
            # IMPORTANT: Only include actual directories, not files
            session_dirs.extend([d for d in found if d.is_dir()])
        
        # Sort by session number
        session_dirs = sorted(session_dirs, key=lambda x: x.name)
        
        if session_dirs:
            print(f"\nFound {len(session_dirs)} session directories:")
            for dir in session_dirs:
                print(f"  - {dir.name}")
        else:
            print(f"\n⚠ No session directories found matching patterns: {patterns}")
        
        return session_dirs
    
    def process_all_sessions(self, session_dirs: List[Path] = None):
        """Process all participant sessions"""
        if session_dirs is None:
            session_dirs = self.find_sessions()
        
        if not session_dirs:
            print("\n✗ No session directories found!")
            print(f"   Searched in: {self.base_dir}")
            self.validate_directory_structure()
            return
        
        print(f"\n{'='*70}")
        print("BATCH PROCESSING PARTICIPANTS")
        print('='*70)
        
        successful = 0
        failed = 0
        
        for session_dir in session_dirs:
            print(f"\nProcessing {session_dir.name}...")
            print("-" * 50)
            session = ParticipantSession(session_dir)
            
            # Load data
            if session.load_behavioral_data():
                session.calculate_metrics()
                self.sessions.append(session)
                
                # Generate individual visualizations
                self._generate_individual_plots(session)
                
                print(f"  ✓ {session_dir.name} processed successfully")
                successful += 1
            else:
                print(f"  ✗ {session_dir.name} failed - no behavioral data")
                failed += 1
        
        print(f"\n{'='*70}")
        print(f"PROCESSING SUMMARY")
        print('='*70)
        print(f"  Successful: {successful}")
        print(f"  Failed:     {failed}")
        print(f"  Total:      {successful + failed}")
        
        # Only generate reports if we have data
        if self.sessions:
            # Generate summary report
            self._generate_summary_report()
            
            # Generate group analysis
            self._generate_group_analysis()
        else:
            print("\n⚠ WARNING: No sessions were successfully processed")
            print("   Check that JSON and XDF files exist in session directories")
    
    def _generate_individual_plots(self, session: ParticipantSession):
        """Generate plots for individual participant"""
        output_subdir = self.output_dir / session.session_name
        output_subdir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Skip if plots already exist and are recent
            existing_plots = list(output_subdir.glob('*.png'))
            if len(existing_plots) >= 2:
                print(f"    ℹ Plots already exist, skipping visualization")
                return
                
            create_comprehensive_visualizations(
                session.statements,
                session.articles,
                session.bias_metrics,
                str(output_subdir)
            )
            print(f"    ✓ Generated visualizations in {output_subdir.name}/")
        except Exception as e:
            print(f"    ⚠ Warning: Could not generate plots: {str(e)[:100]}")
    
    def _generate_summary_report(self):
        """Generate summary report for all participants"""
        if not self.sessions:
            print("\nNo sessions to summarize")
            return
        
        # Collect summaries
        summaries = [session.get_summary() for session in self.sessions]
        self.summary_df = pd.DataFrame(summaries)
        
        # IMPORTANT: Save as 'batch_summary.csv' for compatibility with extract_observed_cb_rate.py
        summary_file = self.output_dir / 'batch_summary.csv'
        self.summary_df.to_csv(summary_file, index=False)
        
        print(f"\n{'='*70}")
        print("BATCH SUMMARY")
        print('='*70)
        print(self.summary_df.to_string(index=False))
        
        # Save as text file for easy viewing
        summary_txt_file = self.output_dir / 'batch_summary.txt'
        with open(summary_txt_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BCI CONFIRMATION BIAS - BATCH SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            f.write(self.summary_df.to_string(index=False))
            f.write("\n\n")
            f.write("="*70 + "\n")
            f.write("GROUP STATISTICS\n")
            f.write("="*70 + "\n")
            
            if 'confirmation_bias_rate' in self.summary_df.columns:
                f.write(f"Mean Confirmation Bias Rate: {self.summary_df['confirmation_bias_rate'].mean():.4f} ({self.summary_df['confirmation_bias_rate'].mean()*100:.2f}%)\n")
                f.write(f"SD Confirmation Bias Rate: {self.summary_df['confirmation_bias_rate'].std():.4f} ({self.summary_df['confirmation_bias_rate'].std()*100:.2f}%)\n")
            
            if 'mean_statement_rt' in self.summary_df.columns:
                f.write(f"Mean Statement RT: {self.summary_df['mean_statement_rt'].mean():.2f}s\n")
            
            if 'mean_article_rt' in self.summary_df.columns:
                f.write(f"Mean Article RT: {self.summary_df['mean_article_rt'].mean():.2f}s\n")
            
            if 'attention_pass_rate' in self.summary_df.columns:
                f.write(f"Mean Attention Pass Rate: {self.summary_df['attention_pass_rate'].mean():.4f} ({self.summary_df['attention_pass_rate'].mean()*100:.2f}%)\n")
        
        # Calculate and display group statistics
        print(f"\n{'='*70}")
        print("GROUP STATISTICS")
        print('='*70)
        
        if 'confirmation_bias_rate' in self.summary_df.columns:
            mean_cb = self.summary_df['confirmation_bias_rate'].mean()
            std_cb = self.summary_df['confirmation_bias_rate'].std()
            print(f"Mean Confirmation Bias Rate: {mean_cb:.4f} ({mean_cb*100:.2f}%)")
            print(f"SD Confirmation Bias Rate: {std_cb:.4f} ({std_cb*100:.2f}%)")
        
        if 'mean_statement_rt' in self.summary_df.columns:
            print(f"Mean Statement RT: {self.summary_df['mean_statement_rt'].mean():.2f}s")
        
        if 'mean_article_rt' in self.summary_df.columns:
            print(f"Mean Article RT: {self.summary_df['mean_article_rt'].mean():.2f}s")
        
        if 'attention_pass_rate' in self.summary_df.columns:
            mean_attn = self.summary_df['attention_pass_rate'].mean()
            print(f"Mean Attention Pass Rate: {mean_attn:.4f} ({mean_attn*100:.2f}%)")
        
        print(f"\n✓ Summary saved to: {summary_file}")
        print(f"✓ Text report saved to: {summary_txt_file}")
    
    def _generate_group_analysis(self):
        """Generate group-level analysis and visualizations"""
        if len(self.sessions) < 2:
            print("\nℹ Need at least 2 participants for group analysis")
            return
        
        print(f"\n{'='*70}")
        print("GROUP ANALYSIS")
        print('='*70)
        
        try:
            # Create group visualization
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Group Analysis - All Participants', fontsize=16, fontweight='bold')
            
            # 1. Confirmation bias rates by participant
            ax1 = axes[0, 0]
            if 'confirmation_bias_rate' in self.summary_df.columns:
                self.summary_df.plot(x='participant_id', y='confirmation_bias_rate', 
                                     kind='bar', ax=ax1, color='#ef4444', alpha=0.7, legend=False)
                ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax1.set_ylabel('Confirmation Bias Rate')
                ax1.set_title('Confirmation Bias by Participant')
                ax1.set_ylim(0, 1)
                ax1.set_xlabel('Participant ID')
            
            # 2. RT comparison
            ax2 = axes[0, 1]
            if 'mean_statement_rt' in self.summary_df.columns and 'mean_article_rt' in self.summary_df.columns:
                rt_data = self.summary_df[['participant_id', 'mean_statement_rt', 'mean_article_rt']]
                rt_data.plot(x='participant_id', kind='bar', ax=ax2, alpha=0.7)
                ax2.set_ylabel('Mean RT (seconds)')
                ax2.set_title('Response Times')
                ax2.legend(['Statements', 'Articles'])
                ax2.set_xlabel('Participant ID')
            
            # 3. Attention rates
            ax3 = axes[0, 2]
            if 'attention_pass_rate' in self.summary_df.columns:
                self.summary_df.plot(x='participant_id', y='attention_pass_rate',
                                     kind='bar', ax=ax3, color='#10b981', alpha=0.7, legend=False)
                ax3.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
                ax3.set_ylabel('Pass Rate')
                ax3.set_title('Attention Check Performance')
                ax3.set_ylim(0, 1.05)
                ax3.legend()
                ax3.set_xlabel('Participant ID')
            
            # 4. Sample sizes
            ax4 = axes[1, 0]
            if 'n_statements' in self.summary_df.columns and 'n_articles' in self.summary_df.columns:
                sample_data = self.summary_df[['participant_id', 'n_statements', 'n_articles']]
                sample_data.plot(x='participant_id', kind='bar', ax=ax4, alpha=0.7)
                ax4.set_ylabel('Count')
                ax4.set_title('Sample Sizes')
                ax4.legend(['Statements', 'Articles'])
                ax4.set_xlabel('Participant ID')
            
            # 5. Bias rates comparison
            ax5 = axes[1, 1]
            if 'confirmation_bias_rate' in self.summary_df.columns and 'disconfirmation_rate' in self.summary_df.columns:
                bias_data = self.summary_df[['participant_id', 'confirmation_bias_rate', 'disconfirmation_rate']]
                bias_data.plot(x='participant_id', kind='bar', ax=ax5, alpha=0.7, color=['#ef4444', '#3b82f6'])
                ax5.set_ylabel('Rate')
                ax5.set_title('Bias Type Rates')
                ax5.legend(['Confirmation', 'Disconfirmation'])
                ax5.set_xlabel('Participant ID')
            
            # 6. Summary text
            ax6 = axes[1, 2]
            ax6.axis('off')
            summary_text = f"Total Participants: {len(self.sessions)}\n"
            summary_text += f"Total Statements: {self.summary_df['n_statements'].sum()}\n"
            summary_text += f"Total Articles: {self.summary_df['n_articles'].sum()}\n\n"
            
            if 'confirmation_bias_rate' in self.summary_df.columns:
                summary_text += f"Mean CB Rate: {self.summary_df['confirmation_bias_rate'].mean():.1%}\n"
                summary_text += f"SD CB Rate: {self.summary_df['confirmation_bias_rate'].std():.1%}\n\n"
            
            if 'attention_pass_rate' in self.summary_df.columns:
                summary_text += f"Mean Attention: {self.summary_df['attention_pass_rate'].mean():.1%}\n"
            
            ax6.text(0.1, 0.8, summary_text, transform=ax6.transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            ax6.set_title('Summary')
            
            plt.tight_layout()
            
            group_plot_file = self.output_dir / 'group_analysis.png'
            plt.savefig(group_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Group analysis saved to: {group_plot_file}")
            
        except Exception as e:
            print(f"  ⚠ Warning: Could not generate group visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def export_for_statistical_analysis(self):
        """Export data in format ready for statistical analysis"""
        if not self.sessions:
            print("\nNo data to export")
            return
        
        print(f"\n{'='*70}")
        print("EXPORTING DETAILED DATA")
        print('='*70)
        
        # Create detailed dataframe
        detailed_data = []
        
        for session in self.sessions:
            # Add statement-level data
            for stmt in session.statements:
                if stmt.attention_check == 'PASS':
                    detailed_data.append({
                        'participant_id': session.participant_id,
                        'session': session.session_name,
                        'phase': 'statement',
                        'item_code': stmt.statement_code,
                        'topic': stmt.topic_code,
                        'agreement': stmt.agreement,
                        'rt': stmt.reaction_time,
                        'attention_check': stmt.attention_check
                    })
            
            # Add article-level data
            bias_lookup = {b['article_code']: b for b in session.bias_metrics.get('detailed_analyses', [])}
            
            for art in session.articles:
                if art.attention_check == 'PASS':
                    bias_info = bias_lookup.get(art.article_code, {})
                    detailed_data.append({
                        'participant_id': session.participant_id,
                        'session': session.session_name,
                        'phase': 'article',
                        'item_code': art.article_code,
                        'topic': art.topic_code,
                        'agreement': art.agreement,
                        'rt': art.reading_time,
                        'attention_check': art.attention_check,
                        'bias_type': bias_info.get('bias_type', 'unknown'),
                        'article_type': bias_info.get('article_type', 'unknown')
                    })
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_file = self.output_dir / 'detailed_behavioral_data.csv'
            detailed_df.to_csv(detailed_file, index=False)
            
            print(f"  ✓ Exported detailed data for {len(self.sessions)} participants")
            print(f"    Total valid responses: {len(detailed_df)}")
            print(f"    File: {detailed_file}")
        else:
            print(f"  ⚠ No valid data to export")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='BCI Batch Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all test_session* folders in current directory
  python bci_batch_process.py
  
  # Process specific sessions
  python bci_batch_process.py --sessions test_session11 test_session12 test_session13
  
  # Use different base directory
  python bci_batch_process.py --base_dir /path/to/data/
  
  # Use different output directory
  python bci_batch_process.py --output_dir ./my_results/
        """
    )
    parser.add_argument('--base_dir', default='./', 
                       help='Base directory containing session folders (default: current directory)')
    parser.add_argument('--output_dir', default='./bci_output/patched/batch_behavioural/',
                       help='Output directory for results (default: ./bci_output/patched/batch_behavioural/)')
    parser.add_argument('--sessions', nargs='+', 
                       help='Specific session directories to process (e.g., test_session11 test_session12)')
    parser.add_argument('--pattern', default=None,
                       help='Pattern for finding session directories (e.g., "test_session*")')
    parser.add_argument('--validate', action='store_true',
                       help='Validate directory structure without processing')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BCIBatchProcessor(args.base_dir, args.output_dir)
    
    # Validation mode
    if args.validate:
        processor.validate_directory_structure()
        return
    
    # Process sessions
    if args.sessions:
        # Process specific sessions
        session_dirs = [processor.base_dir / s for s in args.sessions]
        processor.process_all_sessions(session_dirs)
    elif args.pattern:
        # Use custom pattern
        session_dirs = processor.find_sessions([args.pattern])
        processor.process_all_sessions(session_dirs)
    else:
        # Process all found sessions
        processor.process_all_sessions()
    
    # Export for statistical analysis
    processor.export_for_statistical_analysis()
    
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print('='*70)
    print(f"Results saved to: {processor.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review batch_summary.csv")
    print(f"  2. Run: python extract_observed_cb_rate.py --auto_update")
    print(f"  3. Run: python monte_carlo_UPDATED_v3.py")
    print('='*70)

if __name__ == "__main__":
    main()