"""
Q2: PROPER SELF-DIAGNOSIS ANALYSIS
================================================================================
Analyzing POSTS and COMMENTS separately, then linking them
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

print("=" * 80)
print("Q2: PROPER POST vs COMMENT ANALYSIS")
print("=" * 80)

print("\n[STEP 1: LOAD DATA AND SEPARATE POSTS FROM COMMENTS]")
print("-" * 80)

df = pd.read_csv('all_subreddits_cleaned.csv')
print(f"Total rows (comments): {len(df):,}")

# Extract unique posts
posts_df = df.drop_duplicates(subset='post_id').copy()
print(f"Unique posts: {len(posts_df):,}")
print(f"Average comments per post: {len(df)/len(posts_df):.1f}")

# All comments (keep all rows)
comments_df = df.copy()
print(f"Total comments: {len(comments_df):,}")

print("\n[STEP 2: DEFINE SELF-DIAGNOSIS PATTERNS]")
print("-" * 80)

self_diagnosis_patterns = {
    'hypothesis': [
        r'\bi think (i have|it\'?s|this is|i might have)',
        r'\bi believe (i have|it\'?s|this is)',
        r'\bi suspect',
        r'\bmy guess is',
    ],
    'questioning': [
        r'\bcould this be',
        r'\bis this',
        r'\bdo i have',
        r'\bcould i have',
        r'\bam i having',
        r'\bmight this be',
    ],
    'speculation': [
        r'\bmaybe (i have|it\'?s)',
        r'\bpossibly',
        r'\bmight be',
        r'\bcould be',
    ],
    'certainty': [
        r'\bi have',
        r'\bdiagnosed myself',
        r'\bself-diagnosed',
    ],
}

def detect_self_diagnosis(text):
    """Returns True if text contains self-diagnosis patterns"""
    if pd.isna(text) or text == '':
        return False
    text_lower = str(text).lower()
    for category, patterns in self_diagnosis_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
    return False

def get_diagnosis_type(text):
    """Returns the type of self-diagnosis pattern found"""
    if pd.isna(text) or text == '':
        return None
    text_lower = str(text).lower()
    for category, patterns in self_diagnosis_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return category
    return None

print("Pattern detection functions created")

print("\n[STEP 3: ANALYZE POSTS]")
print("-" * 80)

# Combine post title and text
posts_df['full_text'] = posts_df['post_title'].fillna('') + ' ' + posts_df['post_selftext'].fillna('')

# Detect self-diagnosis in posts
posts_df['has_self_diagnosis'] = posts_df['full_text'].apply(detect_self_diagnosis)
posts_df['diagnosis_type'] = posts_df['full_text'].apply(get_diagnosis_type)

# Count
posts_with_diagnosis = posts_df['has_self_diagnosis'].sum()
posts_without_diagnosis = (~posts_df['has_self_diagnosis']).sum()

print(f"\nPOST ANALYSIS:")
print(f"  Posts WITH self-diagnosis: {posts_with_diagnosis:>5,} ({posts_with_diagnosis/len(posts_df)*100:>5.1f}%)")
print(f"  Posts WITHOUT self-diagnosis: {posts_without_diagnosis:>5,} ({posts_without_diagnosis/len(posts_df)*100:>5.1f}%)")

print(f"\nSelf-diagnosis types in posts:")
diagnosis_types = posts_df[posts_df['has_self_diagnosis']]['diagnosis_type'].value_counts()
for dtype, count in diagnosis_types.items():
    print(f"  {dtype:15s}: {count:>4,} ({count/posts_with_diagnosis*100:>5.1f}%)")

print("\n[STEP 4: ANALYZE COMMENTS]")
print("-" * 80)

# Detect self-diagnosis in comments
comments_df['has_self_diagnosis'] = comments_df['comment_body'].apply(detect_self_diagnosis)
comments_df['diagnosis_type'] = comments_df['comment_body'].apply(get_diagnosis_type)

comments_with_diagnosis = comments_df['has_self_diagnosis'].sum()
comments_without_diagnosis = (~comments_df['has_self_diagnosis']).sum()

print(f"\nCOMMENT ANALYSIS:")
print(f"  Comments WITH self-diagnosis: {comments_with_diagnosis:>7,} ({comments_with_diagnosis/len(comments_df)*100:>5.1f}%)")
print(f"  Comments WITHOUT self-diagnosis: {comments_without_diagnosis:>7,} ({comments_without_diagnosis/len(comments_df)*100:>5.1f}%)")

print(f"\nSelf-diagnosis types in comments:")
comment_diagnosis_types = comments_df[comments_df['has_self_diagnosis']]['diagnosis_type'].value_counts()
for dtype, count in comment_diagnosis_types.items():
    print(f"  {dtype:15s}: {count:>6,} ({count/comments_with_diagnosis*100:>5.1f}%)")

print("\n[STEP 5: LINK POSTS TO COMMENTS]")
print("-" * 80)

# For each post, count how many comments it has with self-diagnosis
post_comment_analysis = []

for post_id in posts_df['post_id'].unique():
    post_row = posts_df[posts_df['post_id'] == post_id].iloc[0]
    post_comments = comments_df[comments_df['post_id'] == post_id]
    
    analysis = {
        'post_id': post_id,
        'subreddit': post_row['subreddit'],
        'post_title': post_row['post_title'][:80],
        'post_has_diagnosis': post_row['has_self_diagnosis'],
        'post_diagnosis_type': post_row['diagnosis_type'],
        'total_comments': len(post_comments),
        'comments_with_diagnosis': post_comments['has_self_diagnosis'].sum(),
        'post_score': post_row['post_score'],
    }
    post_comment_analysis.append(analysis)

analysis_df = pd.DataFrame(post_comment_analysis)

print("\nLINKED ANALYSIS:")
print(f"Total posts analyzed: {len(analysis_df):,}")

print("\n[STEP 6: CATEGORIZE POST-COMMENT PATTERNS]")
print("-" * 80)

# Create categories based on post and comment patterns
analysis_df['pattern'] = 'other'

# Pattern 1: Post has self-diagnosis, comments also discuss diagnosis
mask1 = (analysis_df['post_has_diagnosis']) & (analysis_df['comments_with_diagnosis'] > 0)
analysis_df.loc[mask1, 'pattern'] = 'diagnosis_seeking_with_responses'

# Pattern 2: Post has self-diagnosis, no diagnosis in comments
mask2 = (analysis_df['post_has_diagnosis']) & (analysis_df['comments_with_diagnosis'] == 0)
analysis_df.loc[mask2, 'pattern'] = 'diagnosis_seeking_no_diagnosis_responses'

# Pattern 3: Post has NO self-diagnosis, but comments have diagnosis discussion
mask3 = (~analysis_df['post_has_diagnosis']) & (analysis_df['comments_with_diagnosis'] > 0)
analysis_df.loc[mask3, 'pattern'] = 'no_diagnosis_post_but_diagnosis_in_comments'

# Pattern 4: Neither post nor comments have diagnosis language
mask4 = (~analysis_df['post_has_diagnosis']) & (analysis_df['comments_with_diagnosis'] == 0)
analysis_df.loc[mask4, 'pattern'] = 'no_diagnosis_at_all'

print("\nPOST-COMMENT PATTERNS:")
pattern_counts = analysis_df['pattern'].value_counts()
for pattern, count in pattern_counts.items():
    print(f"  {pattern:45s}: {count:>4,} ({count/len(analysis_df)*100:>5.1f}%)")

print("\n[STEP 7: SUBREDDIT BREAKDOWN]")
print("-" * 80)

for subreddit in analysis_df['subreddit'].unique():
    sub_df = analysis_df[analysis_df['subreddit'] == subreddit]
    posts_with_diag = sub_df['post_has_diagnosis'].sum()
    
    print(f"\nr/{subreddit}:")
    print(f"  Total posts: {len(sub_df):,}")
    print(f"  Posts with self-diagnosis: {posts_with_diag:,} ({posts_with_diag/len(sub_df)*100:.1f}%)")
    print(f"  Pattern breakdown:")
    for pattern, count in sub_df['pattern'].value_counts().items():
        print(f"    {pattern:43s}: {count:>4,} ({count/len(sub_df)*100:>5.1f}%)")

print("\n[STEP 8: ENGAGEMENT ANALYSIS]")
print("-" * 80)

print("\nDo self-diagnosis posts get more engagement?")
with_diag = analysis_df[analysis_df['post_has_diagnosis']]
without_diag = analysis_df[~analysis_df['post_has_diagnosis']]

print(f"\nPosts WITH self-diagnosis:")
print(f"  Average score: {with_diag['post_score'].mean():.1f}")
print(f"  Average comments: {with_diag['total_comments'].mean():.1f}")
print(f"  Average diagnosis responses: {with_diag['comments_with_diagnosis'].mean():.1f}")

print(f"\nPosts WITHOUT self-diagnosis:")
print(f"  Average score: {without_diag['post_score'].mean():.1f}")
print(f"  Average comments: {without_diag['total_comments'].mean():.1f}")
print(f"  Average diagnosis responses: {without_diag['comments_with_diagnosis'].mean():.1f}")

print("\n[STEP 9: EXAMPLES OF EACH PATTERN]")
print("-" * 80)

for pattern in analysis_df['pattern'].unique():
    pattern_posts = analysis_df[analysis_df['pattern'] == pattern].head(3)
    print(f"\n{pattern.upper()}:")
    for idx, row in pattern_posts.iterrows():
        print(f"  [{row['subreddit']}] {row['post_title']}")
        print(f"    Score: {row['post_score']}, Comments: {row['total_comments']}, Diagnosis comments: {row['comments_with_diagnosis']}")

print("\n[STEP 10: SAVE RESULTS]")
print("-" * 80)

# Save post-level analysis
posts_df.to_csv('q2_posts_analyzed.csv', index=False)
print("Saved: q2_posts_analyzed.csv")

# Save linked analysis
analysis_df.to_csv('q2_post_comment_patterns.csv', index=False)
print("Saved: q2_post_comment_patterns.csv")

# Save summary
with open('q2_proper_summary.txt', 'w', encoding='utf-8') as f:
    f.write("Q2: SELF-DIAGNOSIS ANALYSIS - PROPER BREAKDOWN\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("DATASET STRUCTURE\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total unique posts: {len(posts_df):,}\n")
    f.write(f"Total comments: {len(comments_df):,}\n")
    f.write(f"Average comments per post: {len(comments_df)/len(posts_df):.1f}\n\n")
    
    f.write("POST ANALYSIS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Posts with self-diagnosis: {posts_with_diagnosis:,} ({posts_with_diagnosis/len(posts_df)*100:.1f}%)\n")
    f.write(f"Posts without self-diagnosis: {posts_without_diagnosis:,} ({posts_without_diagnosis/len(posts_df)*100:.1f}%)\n\n")
    
    f.write("COMMENT ANALYSIS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Comments with self-diagnosis: {comments_with_diagnosis:,} ({comments_with_diagnosis/len(comments_df)*100:.1f}%)\n")
    f.write(f"Comments without self-diagnosis: {comments_without_diagnosis:,} ({comments_without_diagnosis/len(comments_df)*100:.1f}%)\n\n")
    
    f.write("POST-COMMENT PATTERNS\n")
    f.write("-" * 80 + "\n")
    for pattern, count in pattern_counts.items():
        f.write(f"{pattern:45s}: {count:>4,} ({count/len(analysis_df)*100:>5.1f}%)\n")

print("Saved: q2_proper_summary.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nKEY FINDINGS:")
print(f"1. {posts_with_diagnosis/len(posts_df)*100:.1f}% of POSTS contain self-diagnosis language")
print(f"2. {comments_with_diagnosis/len(comments_df)*100:.1f}% of COMMENTS contain self-diagnosis language")
print(f"3. Most common pattern: {pattern_counts.index[0]}")

