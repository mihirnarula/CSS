import pandas as pd
import numpy as np

print("=" * 80)
print("COMPREHENSIVE DATASET ANALYSIS")
print("=" * 80)

# Read in chunks to analyze the full dataset
chunk_size = 500000
chunks_info = []

print("\nAnalyzing dataset in chunks...")

for i, chunk in enumerate(pd.read_csv('all_subreddits.csv', chunksize=chunk_size)):
    subreddit_counts = chunk['subreddit'].value_counts().to_dict()
    
    chunk_info = {
        'chunk_num': i + 1,
        'rows': len(chunk),
        'subreddits': subreddit_counts,
        'unique_posts': chunk['post_id'].nunique(),
        'unique_comments': chunk['comment_id'].nunique(),
        'comment_rows': chunk['comment_id'].notnull().sum(),
        'post_rows': chunk['comment_id'].isnull().sum()
    }
    chunks_info.append(chunk_info)
    print(f"  Chunk {i+1}: {len(chunk):,} rows processed")

print("\n" + "=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)

total_rows = sum(c['rows'] for c in chunks_info)
total_comment_rows = sum(c['comment_rows'] for c in chunks_info)
total_post_rows = sum(c['post_rows'] for c in chunks_info)

print(f"\nTotal rows: {total_rows:,}")
print(f"Total comment rows: {total_comment_rows:,} ({total_comment_rows/total_rows*100:.2f}%)")
print(f"Total post rows: {total_post_rows:,} ({total_post_rows/total_rows*100:.2f}%)")

print("\n" + "=" * 80)
print("SUBREDDIT DISTRIBUTION (Full Dataset)")
print("=" * 80)

# Aggregate subreddit counts across chunks
all_subreddits = {}
for chunk_info in chunks_info:
    for sub, count in chunk_info['subreddits'].items():
        all_subreddits[sub] = all_subreddits.get(sub, 0) + count

print("\nRows per subreddit:")
for sub, count in sorted(all_subreddits.items(), key=lambda x: x[1], reverse=True):
    print(f"  r/{sub}: {count:,} rows ({count/total_rows*100:.2f}%)")

print("\n" + "=" * 80)
print("UNIQUE COUNTS")
print("=" * 80)

# Need to load full dataset for accurate unique counts
print("\nCalculating unique post IDs across full dataset...")
unique_posts = set()
unique_comments = set()

for chunk in pd.read_csv('all_subreddits.csv', chunksize=chunk_size, usecols=['post_id', 'comment_id']):
    unique_posts.update(chunk['post_id'].unique())
    unique_comments.update(chunk['comment_id'].dropna().unique())

print(f"Unique posts: {len(unique_posts):,}")
print(f"Unique comments: {len(unique_comments):,}")
print(f"Average comments per post: {len(unique_comments)/len(unique_posts):.1f}")

print("\n" + "=" * 80)
print("DATASET STRUCTURE ANALYSIS")
print("=" * 80)

print("\nKey findings:")
print("1. This CSV appears to use a DENORMALIZED structure")
print("2. Each row contains BOTH post metadata AND comment data")
print("3. Post information is repeated for every comment on that post")
print("4. This explains why 'post_id' has fewer unique values than rows")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

