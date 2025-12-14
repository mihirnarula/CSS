"""
CASE STUDY: Deep Analysis of Top 10 Posts
================================================================================
Detailed analysis of high-engagement posts to understand:
- Contradictory advice
- OP follow-up behavior
- Helpfulness of online diagnosis
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from collections import Counter

print("=" * 80)
print("CASE STUDY: DEEP ANALYSIS OF TOP 10 POSTS")
print("=" * 80)

print("\n[STEP 1: SELECT TOP 10 HIGH-ENGAGEMENT POSTS]")
print("-" * 80)

# Load data
df = pd.read_csv('all_subreddits_cleaned.csv', low_memory=False)

# Get posts from diagnosis-seeking subreddits
diagnosis_subs = ['AskDocs', 'HealthAnxiety', 'medicaladvice']
df_diagnosis = df[df['subreddit'].isin(diagnosis_subs)].copy()

# Get unique posts with high comment counts
posts = df_diagnosis.groupby('post_id').agg({
    'post_title': 'first',
    'post_selftext': 'first',
    'subreddit': 'first',
    'post_score': 'first',
    'post_num_comments': 'first',
    'post_created_utc': 'first'
}).reset_index()

# Sort by number of comments
posts_sorted = posts.sort_values('post_num_comments', ascending=False)

# Select top 10
top_10_posts = posts_sorted.head(10)

print(f"Selected top 10 posts with most comments:")
print("-" * 80)
for idx, row in top_10_posts.iterrows():
    print(f"\n{row.name + 1}. [{row['subreddit']}] {row['post_title'][:70]}...")
    print(f"   Comments: {row['post_num_comments']}, Score: {row['post_score']}")

print("\n[STEP 2: DEFINE ANALYSIS FUNCTIONS]")
print("-" * 80)

def classify_comment_type(text):
    """Classify comment into advice categories"""
    if pd.isna(text) or text == '':
        return 'unclear'
    
    text_lower = str(text).lower()
    
    # Emergency
    emergency_words = ['emergency', 'er', '911', 'urgent care', 'immediately', 'right now', 'asap']
    if any(word in text_lower for word in emergency_words):
        return 'emergency'
    
    # Doctor recommendation
    doctor_words = ['see a doctor', 'see doctor', 'call doctor', 'visit doctor', 'make appointment']
    if any(word in text_lower for word in doctor_words):
        return 'see_doctor'
    
    # Reassurance
    reassurance_words = ['don\'t worry', 'you\'re fine', 'nothing serious', 'normal', 'not dangerous']
    if any(word in text_lower for word in reassurance_words):
        return 'reassurance'
    
    # Self-care
    selfcare_words = ['rest', 'drink water', 'take', 'try', 'otc', 'over the counter']
    if any(word in text_lower for word in selfcare_words):
        return 'self_care'
    
    # Question/clarification
    if '?' in text:
        return 'question'
    
    return 'discussion'

def detect_disagreement(text):
    """Detect explicit disagreement markers"""
    if pd.isna(text) or text == '':
        return False
    
    text_lower = str(text).lower()
    disagreement_words = ['no', 'disagree', 'wrong', 'actually', 'not true', 'incorrect', 'i don\'t think']
    
    return any(word in text_lower for word in disagreement_words)

def get_sentiment(text):
    """Get sentiment polarity"""
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity
    except:
        return 0

def detect_confidence(text):
    """Detect confidence level in comment"""
    if pd.isna(text) or text == '':
        return 'medium'
    
    text_lower = str(text).lower()
    
    high_confidence = ['definitely', 'certainly', 'sure', 'absolutely', 'clearly', 'obviously']
    low_confidence = ['maybe', 'might', 'could', 'possibly', 'perhaps', 'not sure']
    
    if any(word in text_lower for word in high_confidence):
        return 'high'
    elif any(word in text_lower for word in low_confidence):
        return 'low'
    else:
        return 'medium'

print("Analysis functions defined")

print("\n[STEP 3: ANALYZE EACH POST IN DETAIL]")
print("-" * 80)

case_studies = []

for post_idx, post_row in top_10_posts.iterrows():
    post_id = post_row['post_id']
    
    print(f"\n{'='*80}")
    print(f"CASE STUDY {post_idx + 1}: {post_row['post_title'][:60]}...")
    print(f"Subreddit: r/{post_row['subreddit']}")
    print('='*80)
    
    # Get all comments for this post
    post_comments = df_diagnosis[df_diagnosis['post_id'] == post_id].copy()
    
    print(f"\nTotal comments: {len(post_comments)}")
    
    # Analyze OP's post
    op_text = str(post_row['post_title']) + ' ' + str(post_row['post_selftext'])
    op_sentiment = get_sentiment(op_text)
    
    print(f"\nOP POST ANALYSIS:")
    print(f"  Length: {len(op_text)} characters")
    print(f"  Sentiment: {op_sentiment:.2f} ({'positive' if op_sentiment > 0.1 else 'negative' if op_sentiment < -0.1 else 'neutral'})")
    
    # Analyze all comments
    post_comments['comment_type'] = post_comments['comment_body'].apply(classify_comment_type)
    post_comments['has_disagreement'] = post_comments['comment_body'].apply(detect_disagreement)
    post_comments['sentiment'] = post_comments['comment_body'].apply(get_sentiment)
    post_comments['confidence'] = post_comments['comment_body'].apply(detect_confidence)
    
    # Comment type distribution
    comment_types = post_comments['comment_type'].value_counts()
    
    print(f"\nCOMMENT TYPES:")
    for ctype, count in comment_types.items():
        print(f"  {ctype:15s}: {count:>3} ({count/len(post_comments)*100:>5.1f}%)")
    
    # Detect contradictions
    has_emergency = (post_comments['comment_type'] == 'emergency').sum()
    has_reassurance = (post_comments['comment_type'] == 'reassurance').sum()
    has_disagreement = post_comments['has_disagreement'].sum()
    
    contradiction_score = 0
    if has_emergency > 0 and has_reassurance > 0:
        contradiction_score = 3  # HIGH - conflicting urgency
    elif has_disagreement > 2:
        contradiction_score = 2  # MODERATE - explicit disagreements
    elif len(comment_types) > 4:
        contradiction_score = 1  # LOW - diverse but not contradictory
    
    print(f"\nCONTRADICTION ANALYSIS:")
    print(f"  Emergency advice: {has_emergency}")
    print(f"  Reassurance: {has_reassurance}")
    print(f"  Explicit disagreements: {has_disagreement}")
    print(f"  Contradiction score: {contradiction_score}/3 ({'HIGH' if contradiction_score == 3 else 'MODERATE' if contradiction_score == 2 else 'LOW' if contradiction_score == 1 else 'NONE'})")
    
    # OP follow-up detection
    op_author = post_comments['comment_author'].iloc[0] if len(post_comments) > 0 else None
    # Note: We don't have OP author in our data structure, so we'll estimate
    
    # Sentiment distribution
    avg_sentiment = post_comments['sentiment'].mean()
    
    print(f"\nSENTIMENT ANALYSIS:")
    print(f"  Average comment sentiment: {avg_sentiment:.2f}")
    print(f"  Positive comments: {(post_comments['sentiment'] > 0.1).sum()}")
    print(f"  Negative comments: {(post_comments['sentiment'] < -0.1).sum()}")
    print(f"  Neutral comments: {((post_comments['sentiment'] >= -0.1) & (post_comments['sentiment'] <= 0.1)).sum()}")
    
    # Confidence distribution
    confidence_dist = post_comments['confidence'].value_counts()
    print(f"\nCONFIDENCE LEVELS:")
    for conf, count in confidence_dist.items():
        print(f"  {conf:10s}: {count:>3}")
    
    # Sample comments
    print(f"\nSAMPLE COMMENTS:")
    for i, (idx, comment) in enumerate(post_comments.head(5).iterrows(), 1):
        comment_preview = str(comment['comment_body'])[:100].replace('\n', ' ')
        print(f"  {i}. [{comment['comment_type']}] {comment_preview}...")
    
    # Store case study data
    case_studies.append({
        'post_id': post_id,
        'post_title': post_row['post_title'],
        'subreddit': post_row['subreddit'],
        'num_comments': len(post_comments),
        'post_score': post_row['post_score'],
        'op_sentiment': op_sentiment,
        'avg_comment_sentiment': avg_sentiment,
        'contradiction_score': contradiction_score,
        'has_emergency': has_emergency,
        'has_reassurance': has_reassurance,
        'has_disagreement': has_disagreement,
        'comment_types': comment_types.to_dict()
    })

print("\n" + "=" * 80)
print("OVERALL FINDINGS ACROSS TOP 10 POSTS")
print("=" * 80)

case_df = pd.DataFrame(case_studies)

print(f"\nContradiction Distribution:")
print(f"  High contradiction (score 3): {(case_df['contradiction_score'] == 3).sum()} posts")
print(f"  Moderate contradiction (score 2): {(case_df['contradiction_score'] == 2).sum()} posts")
print(f"  Low contradiction (score 1): {(case_df['contradiction_score'] == 1).sum()} posts")
print(f"  No contradiction (score 0): {(case_df['contradiction_score'] == 0).sum()} posts")

avg_contradiction = case_df['contradiction_score'].mean()
print(f"\nAverage contradiction score: {avg_contradiction:.2f}/3")

print(f"\nSentiment Patterns:")
print(f"  Average OP sentiment: {case_df['op_sentiment'].mean():.2f}")
print(f"  Average comment sentiment: {case_df['avg_comment_sentiment'].mean():.2f}")

print(f"\nAdvice Patterns:")
print(f"  Posts with emergency advice: {(case_df['has_emergency'] > 0).sum()}/10")
print(f"  Posts with reassurance: {(case_df['has_reassurance'] > 0).sum()}/10")
print(f"  Posts with explicit disagreements: {(case_df['has_disagreement'] > 0).sum()}/10")

# Save results
case_df.to_csv('case_study_top_10_posts.csv', index=False)
print("\nSaved: case_study_top_10_posts.csv")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print(f"""
1. CONTRADICTION RATE: {(case_df['contradiction_score'] >= 2).sum()}/10 posts have significant contradictions

2. SENTIMENT: OPs are {'negative' if case_df['op_sentiment'].mean() < 0 else 'neutral/positive'} 
   Comments are {'more positive' if case_df['avg_comment_sentiment'].mean() > case_df['op_sentiment'].mean() else 'similar'}

3. ADVICE DIVERSITY: Most posts receive multiple types of advice
   - Emergency + Reassurance conflicts are {'common' if (case_df['has_emergency'] > 0).sum() > 5 else 'rare'}

4. HELPFULNESS: {'High' if avg_contradiction < 1.5 else 'Moderate' if avg_contradiction < 2.5 else 'Low'} 
   (based on contradiction level)
""")

print("\n" + "=" * 80)
print("CASE STUDY COMPLETE!")
print("=" * 80)

