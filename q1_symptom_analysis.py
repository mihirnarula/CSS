"""
RESEARCH QUESTION 1: SYMPTOM DESCRIPTION PATTERNS
================================================================================
Q: What are the most common symptoms people describe when seeking medical 
   advice online, and how do description patterns differ across subreddits?

PLAN OF ACTION:
================================================================================

PHASE 1: FEATURE ENGINEERING
-----------------------------
1. Extract symptom-related keywords from posts and comments
2. Create symptom categories (pain, fever, mental health, skin, etc.)
3. Identify body parts mentioned
4. Measure language complexity (readability scores)
5. Detect use of medical vs. layman terminology

PHASE 2: TEXT PROCESSING & NLP
-------------------------------
1. Named Entity Recognition (NER) for medical terms
2. Keyword extraction using TF-IDF
3. Pattern matching for common symptom phrases
4. Part-of-speech tagging for symptom descriptors
5. Create symptom co-occurrence networks

PHASE 3: ANALYSIS TECHNIQUES
-----------------------------
1. Frequency analysis of symptoms by subreddit
2. Word clouds and visualization
3. Statistical tests (chi-square) for subreddit differences
4. Clustering similar symptom descriptions
5. Temporal analysis (symptom trends over time)

PHASE 4: VISUALIZATION
----------------------
1. Bar charts: top symptoms by subreddit
2. Heatmaps: symptom distribution across subreddits
3. Network graphs: symptom co-occurrence
4. Word clouds: dominant terms per subreddit

================================================================================
IMPLEMENTATION START
================================================================================
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("Q1: SYMPTOM DESCRIPTION PATTERNS - ANALYSIS")
print("=" * 80)

print("\n[PHASE 1: DATA LOADING]")
print("-" * 80)

# Load cleaned data in chunks to handle size
print("Loading cleaned dataset...")
df = pd.read_csv('all_subreddits_cleaned.csv')
print(f"Loaded: {len(df):,} rows")
print(f"Columns: {list(df.columns)}")

print("\n[PHASE 2: DEFINING SYMPTOM KEYWORDS]")
print("-" * 80)

# Comprehensive symptom keyword dictionary organized by category
symptom_keywords = {
    'pain': [
        'pain', 'painful', 'ache', 'aching', 'hurt', 'hurts', 'hurting', 'sore',
        'tender', 'throbbing', 'sharp pain', 'dull pain', 'burning pain',
        'stabbing', 'cramping', 'discomfort'
    ],
    'fever': [
        'fever', 'febrile', 'temperature', 'hot', 'chills', 'sweating',
        'night sweats', 'high temp', 'feverish'
    ],
    'respiratory': [
        'cough', 'coughing', 'shortness of breath', 'breathing', 'wheeze',
        'wheezing', 'congestion', 'congested', 'runny nose', 'stuffy',
        'sore throat', 'throat pain', 'difficulty breathing', 'breathless'
    ],
    'digestive': [
        'nausea', 'vomiting', 'diarrhea', 'constipation', 'stomach', 'abdominal',
        'bloating', 'gas', 'indigestion', 'heartburn', 'cramps', 'bowel',
        'appetite', 'digestion'
    ],
    'skin': [
        'rash', 'itchy', 'itching', 'itch', 'bump', 'bumps', 'red', 'redness',
        'swelling', 'swollen', 'bruise', 'bruising', 'discoloration', 'spot',
        'spots', 'acne', 'dry skin', 'irritation'
    ],
    'mental_health': [
        'anxiety', 'anxious', 'depression', 'depressed', 'stress', 'stressed',
        'panic', 'worry', 'worried', 'mood', 'sad', 'sadness', 'fear',
        'scared', 'nervous', 'insomnia', 'sleep problems'
    ],
    'fatigue': [
        'tired', 'fatigue', 'exhausted', 'exhaustion', 'weakness', 'weak',
        'lethargy', 'lethargic', 'energy', 'sleepy', 'drowsy'
    ],
    'neurological': [
        'headache', 'migraine', 'dizziness', 'dizzy', 'vertigo', 'numbness',
        'numb', 'tingling', 'tremor', 'shaking', 'seizure', 'confusion',
        'memory', 'vision', 'blurry'
    ],
    'cardiovascular': [
        'chest pain', 'heart', 'palpitations', 'racing heart', 'blood pressure',
        'hypertension', 'pulse', 'circulation'
    ],
    'musculoskeletal': [
        'joint', 'joints', 'muscle', 'muscles', 'stiff', 'stiffness',
        'sprain', 'strain', 'back pain', 'neck pain', 'shoulder pain'
    ]
}

print("Symptom categories defined:")
for category, keywords in symptom_keywords.items():
    print(f"  {category}: {len(keywords)} keywords")

print("\n[PHASE 3: BODY PARTS DICTIONARY]")
print("-" * 80)

body_parts = [
    'head', 'face', 'eye', 'eyes', 'ear', 'ears', 'nose', 'mouth', 'throat',
    'neck', 'shoulder', 'shoulders', 'chest', 'breast', 'arm', 'arms',
    'hand', 'hands', 'finger', 'fingers', 'back', 'spine', 'stomach',
    'abdomen', 'belly', 'hip', 'hips', 'leg', 'legs', 'knee', 'knees',
    'ankle', 'ankles', 'foot', 'feet', 'toe', 'toes', 'skin', 'heart',
    'lung', 'lungs', 'liver', 'kidney', 'kidneys', 'brain', 'groin',
    'genitals', 'penis', 'vagina', 'testicles'
]

print(f"Body parts tracked: {len(body_parts)} terms")

print("\n[PHASE 4: MEDICAL TERMINOLOGY]")
print("-" * 80)

# Common medical conditions people might self-diagnose
medical_conditions = [
    'covid', 'corona', 'diabetes', 'hypertension', 'cancer', 'tumor',
    'infection', 'flu', 'cold', 'pneumonia', 'bronchitis', 'asthma',
    'allergy', 'allergies', 'eczema', 'psoriasis', 'arthritis', 'lupus',
    'thyroid', 'anemia', 'vitamin deficiency', 'migraine', 'ibs',
    'gerd', 'reflux', 'ulcer', 'hernia', 'hemorrhoid', 'uti',
    'std', 'hiv', 'hepatitis', 'stroke', 'heart attack', 'angina'
]

print(f"Medical conditions tracked: {len(medical_conditions)} terms")

print("\n[PHASE 5: CREATING TEXT EXTRACTION FUNCTION]")
print("-" * 80)

def extract_symptoms_from_text(text, symptom_dict):
    """Extract symptoms from text and count by category"""
    if pd.isna(text) or text == '':
        return {}
    
    text_lower = str(text).lower()
    symptom_counts = {}
    
    for category, keywords in symptom_dict.items():
        count = 0
        for keyword in keywords:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count += len(re.findall(pattern, text_lower))
        if count > 0:
            symptom_counts[category] = count
    
    return symptom_counts

def extract_body_parts(text, body_parts_list):
    """Extract body parts mentioned in text"""
    if pd.isna(text) or text == '':
        return []
    
    text_lower = str(text).lower()
    found_parts = []
    
    for part in body_parts_list:
        pattern = r'\b' + re.escape(part) + r'\b'
        if re.search(pattern, text_lower):
            found_parts.append(part)
    
    return found_parts

def extract_conditions(text, conditions_list):
    """Extract medical conditions mentioned in text"""
    if pd.isna(text) or text == '':
        return []
    
    text_lower = str(text).lower()
    found_conditions = []
    
    for condition in conditions_list:
        pattern = r'\b' + re.escape(condition) + r'\b'
        if re.search(pattern, text_lower):
            found_conditions.append(condition)
    
    return found_conditions

print("Text extraction functions created:")
print("  - extract_symptoms_from_text()")
print("  - extract_body_parts()")
print("  - extract_conditions()")

print("\n[PHASE 6: PROCESSING POSTS - SYMPTOM EXTRACTION]")
print("-" * 80)

print("Combining post title and selftext for analysis...")
df['full_post_text'] = df['post_title'].fillna('') + ' ' + df['post_selftext'].fillna('')

print("\nExtracting symptoms from posts (this may take a minute)...")
# Process a sample first for testing
sample_size = min(50000, len(df))
df_sample = df.head(sample_size).copy()

df_sample['post_symptoms'] = df_sample['full_post_text'].apply(
    lambda x: extract_symptoms_from_text(x, symptom_keywords)
)

df_sample['post_body_parts'] = df_sample['full_post_text'].apply(
    lambda x: extract_body_parts(x, body_parts)
)

df_sample['post_conditions'] = df_sample['full_post_text'].apply(
    lambda x: extract_conditions(x, medical_conditions)
)

print(f"Processed {len(df_sample):,} rows")

# Check extraction results
posts_with_symptoms = df_sample['post_symptoms'].apply(lambda x: len(x) > 0).sum()
posts_with_body_parts = df_sample['post_body_parts'].apply(lambda x: len(x) > 0).sum()
posts_with_conditions = df_sample['post_conditions'].apply(lambda x: len(x) > 0).sum()

print(f"\nExtraction Results:")
print(f"  Posts with symptoms: {posts_with_symptoms:,} ({posts_with_symptoms/len(df_sample)*100:.1f}%)")
print(f"  Posts with body parts: {posts_with_body_parts:,} ({posts_with_body_parts/len(df_sample)*100:.1f}%)")
print(f"  Posts with conditions: {posts_with_conditions:,} ({posts_with_conditions/len(df_sample)*100:.1f}%)")

print("\n[PHASE 7: ANALYZING SYMPTOM PATTERNS]")
print("-" * 80)

# Aggregate symptom counts by category
all_symptoms = Counter()
for symptoms_dict in df_sample['post_symptoms']:
    for category, count in symptoms_dict.items():
        all_symptoms[category] += count

print("\nMost Common Symptom Categories (Overall):")
for category, count in all_symptoms.most_common(10):
    print(f"  {category:20s}: {count:>6,} mentions")

print("\n[PHASE 8: SYMPTOM PATTERNS BY SUBREDDIT]")
print("-" * 80)

# Analyze by subreddit
subreddit_symptoms = {}
for subreddit in df_sample['subreddit'].unique():
    sub_df = df_sample[df_sample['subreddit'] == subreddit]
    sub_symptoms = Counter()
    
    for symptoms_dict in sub_df['post_symptoms']:
        for category, count in symptoms_dict.items():
            sub_symptoms[category] += count
    
    subreddit_symptoms[subreddit] = sub_symptoms

print("\nTop 5 Symptoms by Subreddit:")
for subreddit, symptoms in subreddit_symptoms.items():
    print(f"\nr/{subreddit}:")
    for category, count in symptoms.most_common(5):
        print(f"  {category:20s}: {count:>5,} mentions")

print("\n[PHASE 9: BODY PART ANALYSIS]")
print("-" * 80)

# Count all body parts
all_body_parts = Counter()
for parts_list in df_sample['post_body_parts']:
    for part in parts_list:
        all_body_parts[part] += 1

print("\nMost Mentioned Body Parts:")
for part, count in all_body_parts.most_common(20):
    print(f"  {part:15s}: {count:>5,} mentions")

print("\n[PHASE 10: MEDICAL CONDITIONS ANALYSIS]")
print("-" * 80)

# Count all conditions
all_conditions = Counter()
for conditions_list in df_sample['post_conditions']:
    for condition in conditions_list:
        all_conditions[condition] += 1

print("\nMost Mentioned Medical Conditions:")
for condition, count in all_conditions.most_common(20):
    print(f"  {condition:20s}: {count:>5,} mentions")

print("\n[PHASE 11: SAVING RESULTS]")
print("-" * 80)

# Save processed data
output_file = 'q1_symptom_features.csv'
df_sample.to_csv(output_file, index=False)
print(f"\nProcessed data saved to: {output_file}")

# Save summary statistics
with open('q1_symptom_summary.txt', 'w') as f:
    f.write("RESEARCH QUESTION 1: SYMPTOM ANALYSIS SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("OVERALL SYMPTOM CATEGORIES\n")
    f.write("-" * 80 + "\n")
    for category, count in all_symptoms.most_common():
        f.write(f"{category:20s}: {count:>6,} mentions\n")
    
    f.write("\n\nTOP BODY PARTS MENTIONED\n")
    f.write("-" * 80 + "\n")
    for part, count in all_body_parts.most_common(30):
        f.write(f"{part:15s}: {count:>5,} mentions\n")
    
    f.write("\n\nTOP MEDICAL CONDITIONS\n")
    f.write("-" * 80 + "\n")
    for condition, count in all_conditions.most_common(30):
        f.write(f"{condition:20s}: {count:>5,} mentions\n")
    
    f.write("\n\nSYMPTOMS BY SUBREDDIT\n")
    f.write("-" * 80 + "\n")
    for subreddit, symptoms in subreddit_symptoms.items():
        f.write(f"\nr/{subreddit}:\n")
        for category, count in symptoms.most_common():
            f.write(f"  {category:20s}: {count:>5,} mentions\n")

print("Summary saved to: q1_symptom_summary.txt")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE!")
print("=" * 80)
print("\nNEXT STEPS:")
print("1. Create visualizations (bar charts, heatmaps, word clouds)")
print("2. Statistical testing for subreddit differences")
print("3. Advanced NLP (TF-IDF, topic modeling)")
print("4. Temporal analysis of symptom trends")

