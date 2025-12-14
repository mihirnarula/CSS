import pandas as pd

print("=" * 80)
print("CLEANED DATASET SUMMARY")
print("=" * 80)

# Load cleaned data
df = pd.read_csv('all_subreddits_cleaned.csv', nrows=50000)

print("\n[WHAT WE CLEANED]")
print("-" * 80)
print("1. Removed 15,105 deleted/removed comments (4.60%)")
print("2. Removed 30 rows with null comment bodies")
print("3. Removed 760 empty/very short comments (<3 chars)")
print("4. Cleaned whitespace from text columns")
print("5. Converted dates to proper datetime format")
print("\nTotal removed: 15,895 rows (4.84%)")
print("Final dataset: 312,342 rows")

print("\n[DATASET COMPOSITION]")
print("-" * 80)
print(f"Unique posts: 3,573")
print(f"Total comments: 312,342")
print(f"Average comments per post: {312342/3573:.1f}")
print(f"Date range: 2011-09-19 to 2025-12-11 (14+ years)")

print("\n[SUBREDDIT BREAKDOWN]")
print("-" * 80)
subreddit_data = [
    ('medicine', 110660, 35.43),
    ('Health', 73946, 23.68),
    ('AskDocs', 57748, 18.49),
    ('biohackers', 42344, 13.55),
    ('HealthAnxiety', 27310, 8.74),
    ('medicaladvice', 334, 0.11)
]

for sub, count, pct in subreddit_data:
    print(f"r/{sub:20s} {count:>7,} rows ({pct:>5.2f}%)")

print("\n[NEXT STEPS FOR ANALYSIS]")
print("-" * 80)
print("Now we can answer the project research questions:")
print("")
print("1. How do people describe their symptoms and self-diagnose online?")
print("   -> Analyze post titles and comment text for symptom patterns")
print("")
print("2. What emotions dominate these posts?")
print("   -> Sentiment analysis on comments")
print("")
print("3. What health issues are most commonly self-diagnosed?")
print("   -> Topic modeling and keyword extraction")
print("")
print("4. Can AI/NLP identify self-diagnosis behavior?")
print("   -> Build classification models")

print("\n" + "=" * 80)

