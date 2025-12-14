import pandas as pd
import numpy as np

print("=" * 80)
print("STEP 1: Loading CSV with sample (first 100,000 rows)")
print("=" * 80)

# Load a sample first to understand the data
df_sample = pd.read_csv('all_subreddits.csv', nrows=100000)

print(f"\nSample loaded: {len(df_sample):,} rows")
print(f"\nColumn names and types:")
print(df_sample.dtypes)

print("\n" + "=" * 80)
print("STEP 2: Basic Statistics")
print("=" * 80)

print(f"\nDataset shape: {df_sample.shape}")
print(f"Memory usage: {df_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "=" * 80)
print("STEP 3: Column Analysis")
print("=" * 80)

for col in df_sample.columns:
    null_count = df_sample[col].isnull().sum()
    null_pct = (null_count / len(df_sample)) * 100
    print(f"\n{col}:")
    print(f"  - Null values: {null_count:,} ({null_pct:.2f}%)")
    print(f"  - Unique values: {df_sample[col].nunique():,}")
    if col in ['subreddit', 'post_id', 'comment_author']:
        print(f"  - Sample values: {df_sample[col].value_counts().head(3).to_dict()}")

print("\n" + "=" * 80)
print("STEP 4: Subreddit Distribution")
print("=" * 80)

print("\nPosts per subreddit:")
subreddit_counts = df_sample['subreddit'].value_counts()
for sub, count in subreddit_counts.items():
    print(f"  r/{sub}: {count:,} rows ({count/len(df_sample)*100:.2f}%)")

print("\n" + "=" * 80)
print("STEP 5: Post vs Comment Rows")
print("=" * 80)

# A row is a post if comment_id is null
posts_mask = df_sample['comment_id'].isnull()
comments_mask = df_sample['comment_id'].notnull()

print(f"\nPost rows: {posts_mask.sum():,} ({posts_mask.sum()/len(df_sample)*100:.2f}%)")
print(f"Comment rows: {comments_mask.sum():,} ({comments_mask.sum()/len(df_sample)*100:.2f}%)")

print("\n" + "=" * 80)
print("STEP 6: Data Quality Issues")
print("=" * 80)

# Check for empty or deleted content
print("\nChecking content quality:")
if 'post_selftext' in df_sample.columns:
    empty_posts = df_sample['post_selftext'].isin(['', '[deleted]', '[removed]']).sum()
    print(f"  - Empty/deleted post texts: {empty_posts:,}")

if 'comment_body' in df_sample.columns:
    empty_comments = df_sample['comment_body'].isin(['', '[deleted]', '[removed]']).sum()
    print(f"  - Empty/deleted comments: {empty_comments:,}")

print("\n" + "=" * 80)
print("STEP 7: Sample Data Preview")
print("=" * 80)

posts_df = df_sample[df_sample['comment_id'].isnull()]
comments_df = df_sample[df_sample['comment_id'].notnull()]

if len(posts_df) > 0:
    print("\n--- First Post Example ---")
    first_post = posts_df.iloc[0]
    print(f"Subreddit: r/{first_post['subreddit']}")
    print(f"Title: {first_post['post_title'][:100]}...")
    print(f"Score: {first_post['post_score']}")
    print(f"Comments: {first_post['post_num_comments']}")
else:
    print("\n--- No Post Rows in Sample ---")

if len(comments_df) > 0:
    print("\n--- First Comment Example ---")
    first_comment = comments_df.iloc[0]
    print(f"Subreddit: r/{first_comment['subreddit']}")
    print(f"Post ID: {first_comment['post_id']}")
    print(f"Post Title: {first_comment['post_title'][:80]}...")
    print(f"Comment ID: {first_comment['comment_id']}")
    print(f"Author: {first_comment['comment_author']}")
    print(f"Score: {first_comment['comment_score']}")
    print(f"Depth: {first_comment['depth']}")
    print(f"Body preview: {str(first_comment['comment_body'])[:150]}...")
else:
    print("\n--- No Comment Rows in Sample ---")

print("\n" + "=" * 80)
print("STEP 8: Text Length Statistics")
print("=" * 80)

df_sample['post_text_length'] = df_sample['post_selftext'].fillna('').astype(str).str.len()
df_sample['comment_text_length'] = df_sample['comment_body'].fillna('').astype(str).str.len()

print("\nPost text length statistics:")
print(df_sample['post_text_length'].describe())

print("\nComment text length statistics:")
print(df_sample['comment_text_length'].describe())

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

