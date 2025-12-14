"""
RESEARCH QUESTION 8: SUBREDDIT FUNCTIONAL DIFFERENCES
================================================================================
Comprehensive comparison of user behaviors across medical subreddits
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("=" * 80)
print("Q8: SUBREDDIT FUNCTIONAL DIFFERENCES ANALYSIS")
print("=" * 80)

print("\n[STEP 1: LOAD DATA]")
print("-" * 80)

# Load full dataset
df = pd.read_csv('all_subreddits_cleaned.csv', low_memory=False)
print(f"Total rows (comments): {len(df):,}")

# Get unique posts
posts_df = df.drop_duplicates(subset='post_id').copy()
print(f"Unique posts: {len(posts_df):,}")

# Get all comments
comments_df = df.copy()
print(f"Total comments: {len(comments_df):,}")

print("\nSubreddits in dataset:")
for sub, count in posts_df['subreddit'].value_counts().items():
    print(f"  r/{sub}: {count:,} posts")

print("\n" + "=" * 80)
print("PART 1: POST CHARACTERISTICS BY SUBREDDIT")
print("=" * 80)

# Calculate post characteristics
posts_df['title_length'] = posts_df['post_title'].fillna('').str.len()
posts_df['body_length'] = posts_df['post_selftext'].fillna('').str.len()
posts_df['total_text_length'] = posts_df['title_length'] + posts_df['body_length']
posts_df['has_body'] = posts_df['body_length'] > 0
posts_df['is_question'] = posts_df['post_title'].str.contains(r'\?', na=False)

print("\nPost Length Statistics by Subreddit:")
print("-" * 80)
print(f"{'Subreddit':<20} {'Avg Title':<12} {'Avg Body':<12} {'Avg Total':<12} {'% w/ Body':<12}")
print("-" * 80)

for subreddit in sorted(posts_df['subreddit'].unique()):
    sub_posts = posts_df[posts_df['subreddit'] == subreddit]
    avg_title = sub_posts['title_length'].mean()
    avg_body = sub_posts['body_length'].mean()
    avg_total = sub_posts['total_text_length'].mean()
    pct_body = sub_posts['has_body'].mean() * 100
    
    print(f"r/{subreddit:<18} {avg_title:>10.0f}  {avg_body:>10.0f}  {avg_total:>10.0f}  {pct_body:>10.1f}%")

print("\n" + "=" * 80)
print("PART 2: ENGAGEMENT METRICS BY SUBREDDIT")
print("=" * 80)

# Group comments by post to get comment counts
post_comment_counts = comments_df.groupby('post_id').size().reset_index(name='actual_comment_count')
posts_df = posts_df.merge(post_comment_counts, on='post_id', how='left')
posts_df['actual_comment_count'] = posts_df['actual_comment_count'].fillna(0)

print("\nEngagement Statistics by Subreddit:")
print("-" * 80)
print(f"{'Subreddit':<20} {'Avg Score':<12} {'Med Score':<12} {'Avg Comments':<15} {'Response Rate':<15}")
print("-" * 80)

engagement_stats = []

for subreddit in sorted(posts_df['subreddit'].unique()):
    sub_posts = posts_df[posts_df['subreddit'] == subreddit]
    
    avg_score = sub_posts['post_score'].mean()
    med_score = sub_posts['post_score'].median()
    avg_comments = sub_posts['actual_comment_count'].mean()
    response_rate = (sub_posts['actual_comment_count'] > 0).mean() * 100
    
    print(f"r/{subreddit:<18} {avg_score:>10.1f}  {med_score:>10.0f}  {avg_comments:>13.1f}  {response_rate:>13.1f}%")
    
    engagement_stats.append({
        'subreddit': subreddit,
        'avg_score': avg_score,
        'med_score': med_score,
        'avg_comments': avg_comments,
        'response_rate': response_rate
    })

engagement_df = pd.DataFrame(engagement_stats)

print("\n" + "=" * 80)
print("PART 3: COMMENT CHARACTERISTICS BY SUBREDDIT")
print("=" * 80)

comments_df['comment_length'] = comments_df['comment_body'].fillna('').str.len()

print("\nComment Statistics by Subreddit:")
print("-" * 80)
print(f"{'Subreddit':<20} {'Total Comments':<15} {'Avg Length':<12} {'Med Length':<12}")
print("-" * 80)

for subreddit in sorted(comments_df['subreddit'].unique()):
    sub_comments = comments_df[comments_df['subreddit'] == subreddit]
    
    total_comments = len(sub_comments)
    avg_length = sub_comments['comment_length'].mean()
    med_length = sub_comments['comment_length'].median()
    
    print(f"r/{subreddit:<18} {total_comments:>13,}  {avg_length:>10.1f}  {med_length:>10.0f}")

print("\n" + "=" * 80)
print("PART 4: LANGUAGE ANALYSIS - MEDICAL TERMINOLOGY")
print("=" * 80)

# Define medical terms
medical_terms = [
    'diagnosis', 'symptoms', 'treatment', 'chronic', 'acute', 'syndrome',
    'disorder', 'condition', 'disease', 'infection', 'inflammation',
    'patient', 'clinical', 'medical', 'doctor', 'physician', 'hospital',
    'medication', 'prescription', 'therapy', 'surgery', 'procedure'
]

def count_medical_terms(text):
    """Count medical terminology in text"""
    if pd.isna(text) or text == '':
        return 0
    text_lower = str(text).lower()
    count = 0
    for term in medical_terms:
        count += len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
    return count

def calculate_medical_density(text):
    """Calculate medical terms per 100 words"""
    if pd.isna(text) or text == '':
        return 0
    words = len(str(text).split())
    if words == 0:
        return 0
    med_count = count_medical_terms(text)
    return (med_count / words) * 100

print("Analyzing medical terminology usage...")
posts_df['full_text'] = posts_df['post_title'].fillna('') + ' ' + posts_df['post_selftext'].fillna('')
posts_df['medical_term_count'] = posts_df['full_text'].apply(count_medical_terms)
posts_df['medical_density'] = posts_df['full_text'].apply(calculate_medical_density)

print("\nMedical Terminology Usage by Subreddit:")
print("-" * 80)
print(f"{'Subreddit':<20} {'Avg Med Terms':<15} {'Med Density':<15} {'% w/ Med Terms':<15}")
print("-" * 80)

for subreddit in sorted(posts_df['subreddit'].unique()):
    sub_posts = posts_df[posts_df['subreddit'] == subreddit]
    
    avg_terms = sub_posts['medical_term_count'].mean()
    avg_density = sub_posts['medical_density'].mean()
    pct_with_terms = (sub_posts['medical_term_count'] > 0).mean() * 100
    
    print(f"r/{subreddit:<18} {avg_terms:>13.2f}  {avg_density:>13.2f}%  {pct_with_terms:>13.1f}%")

print("\n" + "=" * 80)
print("PART 5: EMOTIONAL LANGUAGE ANALYSIS")
print("=" * 80)

# Define emotional markers
anxiety_words = ['worried', 'anxious', 'scared', 'fear', 'panic', 'nervous', 'concerned', 'stress']
urgency_words = ['urgent', 'emergency', 'asap', 'immediately', 'help', 'please help', 'desperate']
uncertainty_words = ['unsure', 'confused', 'not sure', 'don\'t know', 'uncertain', 'wondering']

def count_emotional_words(text, word_list):
    """Count emotional words in text"""
    if pd.isna(text) or text == '':
        return 0
    text_lower = str(text).lower()
    count = 0
    for word in word_list:
        count += len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
    return count

posts_df['anxiety_words'] = posts_df['full_text'].apply(lambda x: count_emotional_words(x, anxiety_words))
posts_df['urgency_words'] = posts_df['full_text'].apply(lambda x: count_emotional_words(x, urgency_words))
posts_df['uncertainty_words'] = posts_df['full_text'].apply(lambda x: count_emotional_words(x, uncertainty_words))

print("\nEmotional Language by Subreddit:")
print("-" * 80)
print(f"{'Subreddit':<20} {'Anxiety':<12} {'Urgency':<12} {'Uncertainty':<12}")
print("-" * 80)

for subreddit in sorted(posts_df['subreddit'].unique()):
    sub_posts = posts_df[posts_df['subreddit'] == subreddit]
    
    avg_anxiety = sub_posts['anxiety_words'].mean()
    avg_urgency = sub_posts['urgency_words'].mean()
    avg_uncertainty = sub_posts['uncertainty_words'].mean()
    
    print(f"r/{subreddit:<18} {avg_anxiety:>10.2f}  {avg_urgency:>10.2f}  {avg_uncertainty:>10.2f}")

print("\n" + "=" * 80)
print("PART 6: QUESTION VS STATEMENT POSTS")
print("=" * 80)

print("\nPost Types by Subreddit:")
print("-" * 80)
print(f"{'Subreddit':<20} {'% Questions':<15} {'Avg ? Marks':<15}")
print("-" * 80)

for subreddit in sorted(posts_df['subreddit'].unique()):
    sub_posts = posts_df[posts_df['subreddit'] == subreddit]
    
    pct_questions = sub_posts['is_question'].mean() * 100
    avg_question_marks = sub_posts['full_text'].str.count(r'\?').mean()
    
    print(f"r/{subreddit:<18} {pct_questions:>13.1f}%  {avg_question_marks:>13.2f}")

print("\n" + "=" * 80)
print("PART 7: SELF-DIAGNOSIS PREVALENCE (from Q2)")
print("=" * 80)

# Load Q2 results
try:
    q2_results = pd.read_csv('q2_posts_analyzed.csv')
    
    print("\nSelf-Diagnosis Rates by Subreddit:")
    print("-" * 80)
    print(f"{'Subreddit':<20} {'Posts':<10} {'w/ Self-Dx':<12} {'% Self-Dx':<12}")
    print("-" * 80)
    
    for subreddit in sorted(q2_results['subreddit'].unique()):
        sub_posts = q2_results[q2_results['subreddit'] == subreddit]
        total = len(sub_posts)
        with_dx = sub_posts['has_self_diagnosis'].sum()
        pct_dx = (with_dx / total) * 100 if total > 0 else 0
        
        print(f"r/{subreddit:<18} {total:>8,}  {with_dx:>10,}  {pct_dx:>10.1f}%")
except:
    print("Q2 results not found, skipping self-diagnosis analysis")

print("\n" + "=" * 80)
print("PART 8: STATISTICAL SIGNIFICANCE TESTS")
print("=" * 80)

print("\nTesting if differences are statistically significant...")

# ANOVA test for post scores across subreddits
subreddit_groups = [posts_df[posts_df['subreddit'] == sub]['post_score'].dropna() 
                    for sub in posts_df['subreddit'].unique()]
f_stat, p_value = stats.f_oneway(*subreddit_groups)

print(f"\nPost Scores across subreddits:")
print(f"  F-statistic: {f_stat:.2f}")
print(f"  P-value: {p_value:.4e}")
print(f"  Significant difference: {'YES' if p_value < 0.05 else 'NO'}")

# ANOVA test for comment counts
comment_groups = [posts_df[posts_df['subreddit'] == sub]['actual_comment_count'].dropna() 
                  for sub in posts_df['subreddit'].unique()]
f_stat, p_value = stats.f_oneway(*comment_groups)

print(f"\nComment counts across subreddits:")
print(f"  F-statistic: {f_stat:.2f}")
print(f"  P-value: {p_value:.4e}")
print(f"  Significant difference: {'YES' if p_value < 0.05 else 'NO'}")

# ANOVA test for medical terminology
med_term_groups = [posts_df[posts_df['subreddit'] == sub]['medical_term_count'].dropna() 
                   for sub in posts_df['subreddit'].unique()]
f_stat, p_value = stats.f_oneway(*med_term_groups)

print(f"\nMedical terminology usage across subreddits:")
print(f"  F-statistic: {f_stat:.2f}")
print(f"  P-value: {p_value:.4e}")
print(f"  Significant difference: {'YES' if p_value < 0.05 else 'NO'}")

print("\n" + "=" * 80)
print("PART 9: SUBREDDIT PROFILES SUMMARY")
print("=" * 80)

print("\nCreating comprehensive subreddit profiles...")

profiles = []
for subreddit in sorted(posts_df['subreddit'].unique()):
    sub_posts = posts_df[posts_df['subreddit'] == subreddit]
    sub_comments = comments_df[comments_df['subreddit'] == subreddit]
    
    profile = {
        'subreddit': subreddit,
        'total_posts': len(sub_posts),
        'avg_post_length': sub_posts['total_text_length'].mean(),
        'avg_score': sub_posts['post_score'].mean(),
        'avg_comments': sub_posts['actual_comment_count'].mean(),
        'pct_questions': sub_posts['is_question'].mean() * 100,
        'medical_density': sub_posts['medical_density'].mean(),
        'anxiety_score': sub_posts['anxiety_words'].mean(),
        'urgency_score': sub_posts['urgency_words'].mean(),
        'avg_comment_length': sub_comments['comment_length'].mean(),
    }
    profiles.append(profile)

profiles_df = pd.DataFrame(profiles)

print("\nSubreddit Profiles:")
print("-" * 80)
for idx, row in profiles_df.iterrows():
    print(f"\nr/{row['subreddit']}:")
    print(f"  Posts: {row['total_posts']:,}")
    print(f"  Avg post length: {row['avg_post_length']:.0f} chars")
    print(f"  Avg score: {row['avg_score']:.1f}")
    print(f"  Avg comments: {row['avg_comments']:.1f}")
    print(f"  Questions: {row['pct_questions']:.1f}%")
    print(f"  Medical density: {row['medical_density']:.2f}%")
    print(f"  Anxiety level: {row['anxiety_score']:.2f}")
    print(f"  Urgency level: {row['urgency_score']:.2f}")

print("\n" + "=" * 80)
print("PART 10: SAVING RESULTS")
print("=" * 80)

# Save detailed post analysis
posts_df.to_csv('q8_posts_with_features.csv', index=False)
print("Saved: q8_posts_with_features.csv")

# Save subreddit profiles
profiles_df.to_csv('q8_subreddit_profiles.csv', index=False)
print("Saved: q8_subreddit_profiles.csv")

# Save engagement stats
engagement_df.to_csv('q8_engagement_stats.csv', index=False)
print("Saved: q8_engagement_stats.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

