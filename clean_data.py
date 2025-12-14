import pandas as pd
import numpy as np

print("=" * 80)
print("DATA CLEANING - STEP BY STEP")
print("=" * 80)

print("\n[STEP 1] Loading the dataset...")
df = pd.read_csv('all_subreddits.csv')
print(f"Original dataset: {len(df):,} rows, {df.shape[1]} columns")

print("\n" + "=" * 80)
print("CURRENT DATA ISSUES TO ADDRESS:")
print("=" * 80)
print("\n1. Missing subreddits from project scope (r/medical, r/DiagnoseMe)")
print("2. Extra subreddit not in project (r/HealthAnxiety)")
print("3. Deleted/removed content: [deleted], [removed]")
print("4. Missing comment authors: 14.9% null")
print("5. Missing post selftext: 28.78% null")
print("6. Duplicate post information (denormalized structure)")
print("7. Empty comment bodies")

print("\n" + "=" * 80)
print("[STEP 2] Checking for deleted/removed content...")
print("=" * 80)

# Check deleted/removed comments
deleted_comments = df['comment_body'].isin(['[deleted]', '[removed]'])
print(f"\nDeleted/removed comments: {deleted_comments.sum():,} ({deleted_comments.sum()/len(df)*100:.2f}%)")

# Check deleted/removed posts
deleted_posts = df['post_selftext'].isin(['[deleted]', '[removed]'])
print(f"Deleted/removed posts: {deleted_posts.sum():,} ({deleted_posts.sum()/len(df)*100:.2f}%)")

print("\n" + "=" * 80)
print("[STEP 3] Removing deleted/removed content...")
print("=" * 80)

before_count = len(df)
df_clean = df[~deleted_comments & ~deleted_posts].copy()
removed_count = before_count - len(df_clean)
print(f"Removed {removed_count:,} rows with deleted/removed content")
print(f"Remaining: {len(df_clean):,} rows")

print("\n" + "=" * 80)
print("[STEP 4] Handling missing/null values...")
print("=" * 80)

print("\nNull values per column:")
for col in df_clean.columns:
    null_count = df_clean[col].isnull().sum()
    if null_count > 0:
        print(f"  {col}: {null_count:,} ({null_count/len(df_clean)*100:.2f}%)")

# Remove rows where comment_body is null (essential for analysis)
before_count = len(df_clean)
df_clean = df_clean[df_clean['comment_body'].notnull()].copy()
removed_count = before_count - len(df_clean)
print(f"\nRemoved {removed_count:,} rows with null comment_body")
print(f"Remaining: {len(df_clean):,} rows")

print("\n" + "=" * 80)
print("[STEP 5] Removing empty comments...")
print("=" * 80)

# Remove empty or very short comments (< 3 characters)
df_clean['comment_length'] = df_clean['comment_body'].str.len()
empty_comments = df_clean['comment_length'] < 3
print(f"Empty/too short comments: {empty_comments.sum():,}")

before_count = len(df_clean)
df_clean = df_clean[~empty_comments].copy()
removed_count = before_count - len(df_clean)
print(f"Removed {removed_count:,} rows")
print(f"Remaining: {len(df_clean):,} rows")

print("\n" + "=" * 80)
print("[STEP 6] Analyzing subreddit coverage...")
print("=" * 80)

print("\nCurrent subreddits in dataset:")
for sub, count in df_clean['subreddit'].value_counts().items():
    print(f"  r/{sub}: {count:,} rows")

print("\nExpected subreddits from project:")
expected_subs = ['medicine', 'AskDocs', 'Health', 'biohackers', 'medical', 'medicaladvice', 'DiagnoseMe']
print(f"  {expected_subs}")

print("\nMissing from dataset: r/medical, r/DiagnoseMe")
print("Extra in dataset: r/HealthAnxiety")

print("\n" + "=" * 80)
print("[STEP 7] Converting date columns to datetime...")
print("=" * 80)

df_clean['post_created_utc'] = pd.to_datetime(df_clean['post_created_utc'])
df_clean['comment_created_utc'] = pd.to_datetime(df_clean['comment_created_utc'])
print("Converted post_created_utc and comment_created_utc to datetime")

print("\nDate range:")
print(f"  Posts: {df_clean['post_created_utc'].min()} to {df_clean['post_created_utc'].max()}")
print(f"  Comments: {df_clean['comment_created_utc'].min()} to {df_clean['comment_created_utc'].max()}")

print("\n" + "=" * 80)
print("[STEP 8] Cleaning text columns...")
print("=" * 80)

# Remove extra whitespace
df_clean['comment_body'] = df_clean['comment_body'].str.strip()
df_clean['post_title'] = df_clean['post_title'].str.strip()
df_clean['post_selftext'] = df_clean['post_selftext'].fillna('').str.strip()

print("Cleaned whitespace from text columns")

print("\n" + "=" * 80)
print("[STEP 9] Final statistics...")
print("=" * 80)

print(f"\nFinal cleaned dataset:")
print(f"  Total rows: {len(df_clean):,}")
print(f"  Unique posts: {df_clean['post_id'].nunique():,}")
print(f"  Unique comments: {df_clean['comment_id'].nunique():,}")
print(f"  Date range: {(df_clean['comment_created_utc'].max() - df_clean['comment_created_utc'].min()).days} days")

print("\nRows removed during cleaning:")
original_rows = len(df)
removed_total = original_rows - len(df_clean)
print(f"  Original: {original_rows:,}")
print(f"  Removed: {removed_total:,} ({removed_total/original_rows*100:.2f}%)")
print(f"  Final: {len(df_clean):,} ({len(df_clean)/original_rows*100:.2f}%)")

print("\n" + "=" * 80)
print("[STEP 10] Saving cleaned data...")
print("=" * 80)

# Drop the temporary comment_length column
df_clean = df_clean.drop('comment_length', axis=1)

output_file = 'all_subreddits_cleaned.csv'
df_clean.to_csv(output_file, index=False)
print(f"\nCleaned data saved to: {output_file}")
print(f"File size: {pd.read_csv(output_file).memory_usage(deep=True).sum() / 1024**2:.2f} MB in memory")

print("\n" + "=" * 80)
print("DATA CLEANING COMPLETE!")
print("=" * 80)

