"""
RESEARCH QUESTION 2: SELF-DIAGNOSIS BEHAVIOR IDENTIFICATION
================================================================================
Q: Can we identify posts/comments that explicitly contain self-diagnosis 
   attempts (e.g., "I think I have...", "Could this be...?")?

PLAN OF ACTION:
================================================================================

PHASE 1: PATTERN DEFINITION
----------------------------
Define linguistic patterns that indicate self-diagnosis:
1. Hypothesis statements: "I think I have...", "I believe it's..."
2. Questioning: "Could this be...?", "Is this...?", "Does this sound like...?"
3. Speculation: "Maybe it's...", "Possibly...", "Might be..."
4. Comparison: "Symptoms match...", "Similar to...", "Looks like..."
5. Certainty statements: "I have...", "I'm sure it's...", "Definitely..."

PHASE 2: FEATURE ENGINEERING
-----------------------------
1. Binary features: has_self_diagnosis, has_question, has_speculation
2. Count features: number of diagnostic phrases, question marks
3. Categorical: diagnosis_type (hypothesis, question, certainty, etc.)
4. Text features: presence of condition names with diagnostic language
5. Confidence level: tentative vs. certain self-diagnosis

PHASE 3: NLP TECHNIQUES
-----------------------
1. Regex pattern matching for diagnostic phrases
2. Dependency parsing for sentence structure
3. Named Entity Recognition for medical conditions
4. Sentiment analysis for confidence/uncertainty
5. Classification model: self-diagnosis vs. information-seeking

PHASE 4: ANALYSIS
-----------------
1. Prevalence of self-diagnosis across subreddits
2. Types of self-diagnosis behavior (tentative vs. certain)
3. Correlation with engagement (do self-diagnosis posts get more responses?)
4. Temporal patterns (has self-diagnosis increased over time?)
5. Success rate (do people get confirmations or corrections?)

================================================================================
IMPLEMENTATION START
================================================================================
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("Q2: SELF-DIAGNOSIS BEHAVIOR IDENTIFICATION")
print("=" * 80)

print("\n[PHASE 1: DEFINING SELF-DIAGNOSIS PATTERNS]")
print("-" * 80)

# Comprehensive self-diagnosis phrase patterns
self_diagnosis_patterns = {
    'hypothesis': [
        r'\bi think (i have|it\'?s|this is|i might have)',
        r'\bi believe (i have|it\'?s|this is|i might have)',
        r'\bi suspect (i have|it\'?s|this is)',
        r'\bmy guess is',
        r'\bi\'?m guessing',
        r'\bpretty sure (i have|it\'?s)',
        r'\bfairly certain',
    ],
    'questioning': [
        r'\bcould this be',
        r'\bis this',
        r'\bwould this be',
        r'\bmight this be',
        r'\bdoes this sound like',
        r'\bdoes this look like',
        r'\bdo i have',
        r'\bam i having',
        r'\bshould i be worried about',
        r'\bcould i have',
        r'\bwhat are the chances',
        r'\bis it possible (i have|that i have)',
    ],
    'speculation': [
        r'\bmaybe (i have|it\'?s)',
        r'\bpossibly',
        r'\bperhaps (i have|it\'?s)',
        r'\bmight be',
        r'\bcould be',
        r'\bseems like',
        r'\bappears to be',
        r'\blooks like (i have|it might be)',
    ],
    'comparison': [
        r'\bsymptoms match',
        r'\bsymptoms (are )?similar to',
        r'\bsame symptoms as',
        r'\bfits the description of',
        r'\bmatches (the )?symptoms of',
        r'\bresembles',
        r'\bsounds like (i have|it could be)',
    ],
    'certainty': [
        r'\bi have',
        r'\bi\'?m sure (i have|it\'?s)',
        r'\bdefinitely (have|is)',
        r'\bdiagnosed myself with',
        r'\bself-diagnosed with',
        r'\bpretty sure i have',
        r'\bconvinced (i have|it\'?s)',
        r'\bno doubt',
    ],
    'seeking_validation': [
        r'\bdo you think (i have|it\'?s)',
        r'\bwhat do you think',
        r'\bdoes anyone else think',
        r'\bam i right (in thinking|to think)',
        r'\bwould you agree',
        r'\bconfirm (my )?suspicion',
        r'\bvalidate',
    ],
    'research_based': [
        r'\bgoogled (my )?symptoms',
        r'\blooked up (my )?symptoms',
        r'\bwebmd says',
        r'\baccording to (google|internet|webmd)',
        r'\bread (online|on the internet) that',
        r'\bfound online',
        r'\bresearched (my )?symptoms',
    ]
}

print("Self-diagnosis pattern categories defined:")
for category, patterns in self_diagnosis_patterns.items():
    print(f"  {category:20s}: {len(patterns)} patterns")

print("\n[PHASE 2: DEFINING MEDICAL CONDITIONS FOR CONTEXT]")
print("-" * 80)

# Expanded medical conditions list
medical_conditions = [
    'covid', 'coronavirus', 'cancer', 'diabetes', 'hypertension', 'pneumonia',
    'bronchitis', 'asthma', 'copd', 'flu', 'influenza', 'cold', 'strep',
    'infection', 'uti', 'std', 'hiv', 'aids', 'hepatitis', 'tuberculosis',
    'arthritis', 'lupus', 'fibromyalgia', 'crohn', 'ibs', 'gerd', 'reflux',
    'ulcer', 'gastritis', 'appendicitis', 'hernia', 'hemorrhoid',
    'eczema', 'psoriasis', 'dermatitis', 'acne', 'rosacea',
    'depression', 'anxiety', 'adhd', 'bipolar', 'schizophrenia', 'ptsd',
    'migraine', 'concussion', 'stroke', 'aneurysm', 'seizure', 'epilepsy',
    'heart attack', 'angina', 'arrhythmia', 'tachycardia',
    'anemia', 'leukemia', 'lymphoma', 'tumor', 'cyst',
    'thyroid', 'hypothyroid', 'hyperthyroid', 'hashimoto',
    'kidney stone', 'kidney disease', 'liver disease', 'cirrhosis',
    'mono', 'mononucleosis', 'meningitis', 'sepsis', 'pneumothorax'
]

print(f"Tracking {len(medical_conditions)} medical conditions")

print("\n[PHASE 3: CREATING DETECTION FUNCTIONS]")
print("-" * 80)

def detect_self_diagnosis(text, patterns_dict):
    """
    Detect self-diagnosis patterns in text
    Returns dict with categories and matched patterns
    """
    if pd.isna(text) or text == '':
        return {}
    
    text_lower = str(text).lower()
    detected = {}
    
    for category, patterns in patterns_dict.items():
        matches = []
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matches.append(pattern)
        
        if matches:
            detected[category] = len(matches)
    
    return detected

def extract_self_diagnosis_context(text, patterns_dict, conditions_list):
    """
    Extract sentences containing self-diagnosis with condition mentions
    """
    if pd.isna(text) or text == '':
        return []
    
    text_lower = str(text).lower()
    sentences = re.split(r'[.!?]+', text_lower)
    
    diagnosis_contexts = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        
        # Check if sentence has diagnosis pattern
        has_pattern = False
        for category, patterns in patterns_dict.items():
            for pattern in patterns:
                if re.search(pattern, sentence):
                    has_pattern = True
                    break
            if has_pattern:
                break
        
        # Check if sentence mentions a condition
        has_condition = False
        mentioned_condition = None
        for condition in conditions_list:
            if re.search(r'\b' + re.escape(condition) + r'\b', sentence):
                has_condition = True
                mentioned_condition = condition
                break
        
        # If both present, store context
        if has_pattern and has_condition:
            diagnosis_contexts.append({
                'sentence': sentence[:200],  # Limit length
                'condition': mentioned_condition
            })
    
    return diagnosis_contexts

def calculate_confidence_score(text):
    """
    Calculate confidence level in self-diagnosis (0-1 scale)
    Based on certainty language
    """
    if pd.isna(text) or text == '':
        return 0.5
    
    text_lower = str(text).lower()
    
    # High confidence markers
    high_conf = ['definitely', 'certain', 'sure', 'convinced', 'no doubt', 'clearly']
    # Low confidence markers
    low_conf = ['maybe', 'possibly', 'perhaps', 'might', 'could', 'not sure', 'unsure']
    # Question markers (uncertainty)
    questions = text_lower.count('?')
    
    high_score = sum(1 for marker in high_conf if marker in text_lower)
    low_score = sum(1 for marker in low_conf if marker in text_lower)
    
    # Calculate score (0.5 is neutral)
    score = 0.5 + (high_score * 0.1) - (low_score * 0.1) - (questions * 0.05)
    return max(0, min(1, score))  # Clamp between 0 and 1

print("Detection functions created:")
print("  - detect_self_diagnosis()")
print("  - extract_self_diagnosis_context()")
print("  - calculate_confidence_score()")

print("\n[PHASE 4: LOADING AND PROCESSING DATA]")
print("-" * 80)

print("Loading cleaned dataset...")
df = pd.read_csv('all_subreddits_cleaned.csv', nrows=100000)
print(f"Loaded: {len(df):,} rows")

# Combine post and comment text for analysis
print("\nCreating combined text fields...")
df['post_full_text'] = df['post_title'].fillna('') + ' ' + df['post_selftext'].fillna('')
df['comment_text'] = df['comment_body'].fillna('')

print("\n[PHASE 5: DETECTING SELF-DIAGNOSIS IN POSTS]")
print("-" * 80)

print("Analyzing posts for self-diagnosis patterns...")
df['post_diagnosis_patterns'] = df['post_full_text'].apply(
    lambda x: detect_self_diagnosis(x, self_diagnosis_patterns)
)

df['post_has_self_diagnosis'] = df['post_diagnosis_patterns'].apply(lambda x: len(x) > 0)
df['post_diagnosis_count'] = df['post_diagnosis_patterns'].apply(lambda x: sum(x.values()))
df['post_confidence_score'] = df['post_full_text'].apply(calculate_confidence_score)

posts_with_diagnosis = df['post_has_self_diagnosis'].sum()
print(f"\nPosts with self-diagnosis patterns: {posts_with_diagnosis:,} ({posts_with_diagnosis/len(df)*100:.1f}%)")

print("\n[PHASE 6: DETECTING SELF-DIAGNOSIS IN COMMENTS]")
print("-" * 80)

print("Analyzing comments for self-diagnosis patterns...")
df['comment_diagnosis_patterns'] = df['comment_text'].apply(
    lambda x: detect_self_diagnosis(x, self_diagnosis_patterns)
)

df['comment_has_self_diagnosis'] = df['comment_diagnosis_patterns'].apply(lambda x: len(x) > 0)
df['comment_diagnosis_count'] = df['comment_diagnosis_patterns'].apply(lambda x: sum(x.values()))
df['comment_confidence_score'] = df['comment_text'].apply(calculate_confidence_score)

comments_with_diagnosis = df['comment_has_self_diagnosis'].sum()
print(f"\nComments with self-diagnosis patterns: {comments_with_diagnosis:,} ({comments_with_diagnosis/len(df)*100:.1f}%)")

print("\n[PHASE 7: ANALYZING PATTERN TYPES]")
print("-" * 80)

# Aggregate pattern types from posts
post_pattern_counts = Counter()
for patterns_dict in df['post_diagnosis_patterns']:
    for category, count in patterns_dict.items():
        post_pattern_counts[category] += count

print("\nSelf-Diagnosis Pattern Types in POSTS:")
for category, count in post_pattern_counts.most_common():
    print(f"  {category:20s}: {count:>6,} occurrences")

# Aggregate pattern types from comments
comment_pattern_counts = Counter()
for patterns_dict in df['comment_diagnosis_patterns']:
    for category, count in patterns_dict.items():
        comment_pattern_counts[category] += count

print("\nSelf-Diagnosis Pattern Types in COMMENTS:")
for category, count in comment_pattern_counts.most_common():
    print(f"  {category:20s}: {count:>6,} occurrences")

print("\n[PHASE 8: SUBREDDIT ANALYSIS]")
print("-" * 80)

print("\nSelf-Diagnosis Prevalence by Subreddit:")
for subreddit in df['subreddit'].unique():
    sub_df = df[df['subreddit'] == subreddit]
    post_rate = sub_df['post_has_self_diagnosis'].mean() * 100
    comment_rate = sub_df['comment_has_self_diagnosis'].mean() * 100
    avg_confidence = sub_df[sub_df['post_has_self_diagnosis']]['post_confidence_score'].mean()
    
    print(f"\nr/{subreddit}:")
    print(f"  Posts with self-diagnosis: {post_rate:5.1f}%")
    print(f"  Comments with self-diagnosis: {comment_rate:5.1f}%")
    print(f"  Average confidence score: {avg_confidence:.2f}")

print("\n[PHASE 9: CONFIDENCE ANALYSIS]")
print("-" * 80)

# Categorize by confidence level
df['confidence_category'] = pd.cut(
    df['post_confidence_score'],
    bins=[0, 0.4, 0.6, 1.0],
    labels=['Low Confidence', 'Medium Confidence', 'High Confidence']
)

print("\nConfidence Distribution in Self-Diagnosis Posts:")
confidence_dist = df[df['post_has_self_diagnosis']]['confidence_category'].value_counts()
for category, count in confidence_dist.items():
    print(f"  {category}: {count:,} ({count/confidence_dist.sum()*100:.1f}%)")

print("\n[PHASE 10: ENGAGEMENT ANALYSIS]")
print("-" * 80)

with_diagnosis = df[df['post_has_self_diagnosis']]
without_diagnosis = df[~df['post_has_self_diagnosis']]

print("\nEngagement Comparison:")
print(f"\nPost Scores:")
print(f"  With self-diagnosis - Mean: {with_diagnosis['post_score'].mean():.1f}, Median: {with_diagnosis['post_score'].median():.1f}")
print(f"  Without self-diagnosis - Mean: {without_diagnosis['post_score'].mean():.1f}, Median: {without_diagnosis['post_score'].median():.1f}")

print(f"\nComment Counts:")
print(f"  With self-diagnosis - Mean: {with_diagnosis['post_num_comments'].mean():.1f}, Median: {with_diagnosis['post_num_comments'].median():.1f}")
print(f"  Without self-diagnosis - Mean: {without_diagnosis['post_num_comments'].mean():.1f}, Median: {without_diagnosis['post_num_comments'].median():.1f}")

print("\n[PHASE 11: EXTRACTING EXAMPLE CONTEXTS]")
print("-" * 80)

print("\nExtracting self-diagnosis examples with conditions...")
df['diagnosis_contexts'] = df['post_full_text'].apply(
    lambda x: extract_self_diagnosis_context(x, self_diagnosis_patterns, medical_conditions)
)

# Get some examples
examples = []
for idx, row in df[df['post_has_self_diagnosis']].head(20).iterrows():
    contexts = row['diagnosis_contexts']
    if contexts:
        for context in contexts[:1]:  # Take first context
            examples.append({
                'subreddit': row['subreddit'],
                'condition': context['condition'],
                'sentence': context['sentence'],
                'confidence': row['post_confidence_score']
            })

print(f"\nFound {len(examples)} example self-diagnosis statements")
print("\nSample Self-Diagnosis Examples:")
for i, ex in enumerate(examples[:5], 1):
    print(f"\n{i}. r/{ex['subreddit']} | Condition: {ex['condition']} | Confidence: {ex['confidence']:.2f}")
    print(f"   \"{ex['sentence'][:150]}...\"")

print("\n[PHASE 12: SAVING RESULTS]")
print("-" * 80)

output_file = 'q2_self_diagnosis_features.csv'
df.to_csv(output_file, index=False)
print(f"\nProcessed data saved to: {output_file}")

# Save summary
with open('q2_self_diagnosis_summary.txt', 'w') as f:
    f.write("RESEARCH QUESTION 2: SELF-DIAGNOSIS DETECTION SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"OVERALL STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total rows analyzed: {len(df):,}\n")
    f.write(f"Posts with self-diagnosis: {posts_with_diagnosis:,} ({posts_with_diagnosis/len(df)*100:.1f}%)\n")
    f.write(f"Comments with self-diagnosis: {comments_with_diagnosis:,} ({comments_with_diagnosis/len(df)*100:.1f}%)\n\n")
    
    f.write("PATTERN TYPES IN POSTS\n")
    f.write("-" * 80 + "\n")
    for category, count in post_pattern_counts.most_common():
        f.write(f"{category:20s}: {count:>6,}\n")
    
    f.write("\n\nPATTERN TYPES IN COMMENTS\n")
    f.write("-" * 80 + "\n")
    for category, count in comment_pattern_counts.most_common():
        f.write(f"{category:20s}: {count:>6,}\n")
    
    f.write("\n\nEXAMPLE SELF-DIAGNOSIS STATEMENTS\n")
    f.write("-" * 80 + "\n")
    for i, ex in enumerate(examples[:20], 1):
        f.write(f"\n{i}. r/{ex['subreddit']} | {ex['condition']} | Confidence: {ex['confidence']:.2f}\n")
        f.write(f"   {ex['sentence']}\n")

print("Summary saved to: q2_self_diagnosis_summary.txt")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE!")
print("=" * 80)
print("\nNEXT STEPS:")
print("1. Create visualizations")
print("2. Build classification model (self-diagnosis vs. information-seeking)")
print("3. Temporal analysis")
print("4. Response analysis (confirmations vs. corrections)")

