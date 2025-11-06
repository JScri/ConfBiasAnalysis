#!/usr/bin/env python3
"""
Check actual rating distributions from Unity JSON output
Works with the actual JSON structure from your experiment
"""
import json
import glob
from collections import Counter
import os

def check_rating_distributions(json_files_pattern):
    """Analyze actual rating distributions from participant data"""
    
    statement_ratings_all = []
    article_ratings_all = []
    statement_ratings_pass = []
    article_ratings_pass = []
    
    files_found = glob.glob(json_files_pattern)
    
    if not files_found:
        print(f"❌ No files found matching pattern: {json_files_pattern}")
        print("\nTrying alternate patterns...")
        
        # Try some common alternatives
        alternates = [
            "./test_session*/participant_*.json",
            "./participant_*.json",
            "participant_*.json"
        ]
        
        for alt in alternates:
            alt_files = glob.glob(alt)
            if alt_files:
                print(f"✓ Found {len(alt_files)} files with pattern: {alt}")
                files_found = alt_files
                break
        
        if not files_found:
            print("\n❌ Could not find any participant JSON files!")
            return
    
    print(f"\n{'='*70}")
    print(f"Found {len(files_found)} JSON files")
    print(f"{'='*70}\n")
    
    for json_file in files_found:
        print(f"Reading: {os.path.basename(json_file)}")
        
        try:
            with open(json_file) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
            continue
        
        # Process statement responses
        responses = data.get('responses', [])
        print(f"  Statement responses: {len(responses)}")
        
        for resp in responses:
            # Get rating (selectedOption is a string "1"-"5")
            rating_str = resp.get('selectedOption', '0')
            try:
                rating = int(rating_str)
                if 1 <= rating <= 5:
                    statement_ratings_all.append(rating)
                    
                    # Check attention
                    attn = resp.get('attentionCheckResponse', 'UNKNOWN')
                    if attn in ['YES', 'NO']:
                        statement_ratings_pass.append(rating)
            except ValueError:
                pass
        
        # Process article responses
        articles = data.get('selectedArticles', [])
        print(f"  Article responses: {len(articles)}")
        
        for art in articles:
            # Get rating (selectedOption is a string "1"-"5")
            rating_str = art.get('selectedOption', '0')
            try:
                rating = int(rating_str)
                if 1 <= rating <= 5:
                    article_ratings_all.append(rating)
                    
                    # Check attention
                    attn = art.get('attentionCheckResponse', 'UNKNOWN')
                    if attn in ['YES', 'NO']:
                        article_ratings_pass.append(rating)
            except ValueError:
                pass
    
    print(f"\n{'='*70}")
    print(f"COLLECTED DATA SUMMARY")
    print(f"{'='*70}")
    print(f"Total statement ratings: {len(statement_ratings_all)}")
    print(f"Total article ratings: {len(article_ratings_all)}")
    print(f"Statement ratings (attention passed): {len(statement_ratings_pass)}")
    print(f"Article ratings (attention passed): {len(article_ratings_pass)}")
    
    if not statement_ratings_all and not article_ratings_all:
        print("\n❌ No rating data found!")
        return
    
    # Calculate distributions
    stmt_counts = Counter(statement_ratings_all)
    art_counts = Counter(article_ratings_all)
    
    print(f"\n{'='*70}")
    print(f"RATING DISTRIBUTIONS (ALL - Use for Monte Carlo)")
    print(f"{'='*70}")
    
    print("\n📊 Statement Ratings (ALL):")
    if statement_ratings_all:
        for rating in [1, 2, 3, 4, 5]:
            count = stmt_counts.get(rating, 0)
            pct = count / len(statement_ratings_all) * 100
            bar = '█' * int(pct / 2)
            print(f"  Rating {rating}: {count:3d} ({pct:5.1f}%) {bar}")
    else:
        print("  No data")
    
    print("\n📊 Article Ratings (ALL):")
    if article_ratings_all:
        for rating in [1, 2, 3, 4, 5]:
            count = art_counts.get(rating, 0)
            pct = count / len(article_ratings_all) * 100
            bar = '█' * int(pct / 2)
            print(f"  Rating {rating}: {count:3d} ({pct:5.1f}%) {bar}")
    else:
        print("  No data")
    
    # Calculate recommended distributions
    print(f"\n{'='*70}")
    print(f"RECOMMENDED MONTE CARLO DISTRIBUTIONS")
    print(f"{'='*70}")
    
    if statement_ratings_all:
        stmt_probs = [stmt_counts.get(r, 0) / len(statement_ratings_all) for r in [1,2,3,4,5]]
        print(f"\n✅ For monte_carlo_CORRECTED.py line 26:")
        print(f"STATEMENT_RATING_PROBS = {[round(p, 3) for p in stmt_probs]}")
    
    if article_ratings_all:
        art_probs = [art_counts.get(r, 0) / len(article_ratings_all) for r in [1,2,3,4,5]]
        print(f"\n✅ For monte_carlo_CORRECTED.py line 27:")
        print(f"ARTICLE_RATING_PROBS = {[round(p, 3) for p in art_probs]}")
    
    # Show attention-passed data for comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON: With Attention Check Filtering")
    print(f"{'='*70}")
    print(f"(For reference only - don't use for Monte Carlo baseline)")
    
    if statement_ratings_pass:
        print(f"\nStatements (passed attention): {len(statement_ratings_pass)}/{len(statement_ratings_all)}")
        stmt_pass_counts = Counter(statement_ratings_pass)
        for rating in [1, 2, 3, 4, 5]:
            count = stmt_pass_counts.get(rating, 0)
            pct = count / len(statement_ratings_pass) * 100
            print(f"  Rating {rating}: {count:3d} ({pct:5.1f}%)")
    
    if article_ratings_pass:
        print(f"\nArticles (passed attention): {len(article_ratings_pass)}/{len(article_ratings_all)}")
        art_pass_counts = Counter(article_ratings_pass)
        for rating in [1, 2, 3, 4, 5]:
            count = art_pass_counts.get(rating, 0)
            pct = count / len(article_ratings_pass) * 100
            print(f"  Rating {rating}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"✅ NEXT STEPS:")
    print(f"{'='*70}")
    print(f"1. Copy the STATEMENT_RATING_PROBS line above")
    print(f"2. Copy the ARTICLE_RATING_PROBS line above")
    print(f"3. Paste into monte_carlo_CORRECTED.py lines 26-27")
    print(f"4. Run: python3 monte_carlo_CORRECTED.py")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    else:
        # Default pattern
        pattern = "./test_session*/participant_*.json"
    
    print(f"\n{'='*70}")
    print(f"RATING DISTRIBUTION CHECKER")
    print(f"{'='*70}")
    print(f"Searching for: {pattern}\n")
    
    check_rating_distributions(pattern)