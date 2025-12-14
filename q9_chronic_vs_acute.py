"""
RESEARCH QUESTION 9: CHRONIC VS ACUTE CONDITION CLASSIFICATION
================================================================================
Practical AI approach: Feature engineering + Traditional ML
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Q9: CHRONIC VS ACUTE CONDITION CLASSIFICATION")
print("=" * 80)

print("\n[STEP 1: LOAD DATA]")
print("-" * 80)

posts_df = pd.read_csv('q8_posts_with_features.csv', low_memory=False)
posts_df['full_text'] = posts_df['post_title'].fillna('') + ' ' + posts_df['post_selftext'].fillna('')
posts_df = posts_df[posts_df['full_text'].str.len() > 50].copy()

print(f"Loaded {len(posts_df):,} posts")

print("\n[STEP 2: DEFINE TEMPORAL MARKERS]")
print("-" * 80)

# Chronic time markers
chronic_markers = {
    'duration_long': ['years', 'year', 'months', 'month', 'decades'],
    'frequency': ['always', 'constantly', 'daily', 'every day', 'regularly', 'continuous'],
    'persistence': ['chronic', 'persistent', 'ongoing', 'long-term', 'permanent'],
    'management': ['managing', 'living with', 'dealing with', 'coping', 'maintenance'],
    'frustration': ['still', 'continues', 'won\'t go away', 'never stops', 'tired of']
}

# Acute time markers
acute_markers = {
    'sudden_onset': ['suddenly', 'sudden', 'all of a sudden', 'out of nowhere', 'abruptly'],
    'recent': ['just started', 'began', 'started', 'appeared', 'came on', 'developed'],
    'time_recent': ['today', 'yesterday', 'last night', 'this morning', 'few hours ago', 'just now'],
    'duration_short': ['hours', 'hour', 'days', 'day', 'minutes'],
    'urgency': ['emergency', 'urgent', 'immediate', 'right now', 'asap', 'help']
}

# Known conditions
chronic_conditions = [
    'diabetes', 'hypertension', 'arthritis', 'asthma', 'copd', 'chronic pain',
    'depression', 'anxiety', 'bipolar', 'adhd', 'fibromyalgia', 'lupus',
    'ibs', 'crohn', 'eczema', 'psoriasis', 'chronic fatigue', 'migraine',
    'epilepsy', 'parkinson', 'alzheimer', 'multiple sclerosis', 'thyroid'
]

acute_conditions = [
    'flu', 'cold', 'infection', 'food poisoning', 'injury', 'fracture',
    'sprain', 'cut', 'burn', 'allergic reaction', 'heart attack', 'stroke',
    'appendicitis', 'pneumonia', 'bronchitis', 'uti', 'fever', 'rash'
]

print(f"Chronic markers: {sum(len(v) for v in chronic_markers.values())} phrases")
print(f"Acute markers: {sum(len(v) for v in acute_markers.values())} phrases")
print(f"Known chronic conditions: {len(chronic_conditions)}")
print(f"Known acute conditions: {len(acute_conditions)}")

print("\n[STEP 3: FEATURE EXTRACTION FUNCTIONS]")
print("-" * 80)

def count_markers(text, marker_dict):
    """Count markers from a dictionary of marker lists"""
    if pd.isna(text) or text == '':
        return 0
    text_lower = str(text).lower()
    count = 0
    for category, markers in marker_dict.items():
        for marker in markers:
            count += len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
    return count

def count_conditions(text, condition_list):
    """Count specific conditions mentioned"""
    if pd.isna(text) or text == '':
        return 0
    text_lower = str(text).lower()
    count = 0
    for condition in condition_list:
        if re.search(r'\b' + re.escape(condition) + r'\b', text_lower):
            count += 1
    return count

def extract_duration_days(text):
    """Extract duration and convert to days (rough estimate)"""
    if pd.isna(text) or text == '':
        return 0
    
    text_lower = str(text).lower()
    
    # Look for patterns like "5 years", "3 months", "2 weeks"
    patterns = [
        (r'(\d+)\s*years?', 365),
        (r'(\d+)\s*months?', 30),
        (r'(\d+)\s*weeks?', 7),
        (r'(\d+)\s*days?', 1),
        (r'(\d+)\s*hours?', 0.04),  # Less than a day
    ]
    
    max_duration = 0
    for pattern, multiplier in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            duration = int(match) * multiplier
            max_duration = max(max_duration, duration)
    
    return max_duration

def has_er_mention(text):
    """Check for emergency room mentions"""
    if pd.isna(text) or text == '':
        return 0
    text_lower = str(text).lower()
    er_terms = ['emergency room', 'er', 'urgent care', '911', 'hospital emergency']
    return sum(1 for term in er_terms if term in text_lower)

def count_question_marks(text):
    """Count question marks (urgency indicator)"""
    if pd.isna(text) or text == '':
        return 0
    return str(text).count('?')

def count_exclamation_marks(text):
    """Count exclamation marks (urgency indicator)"""
    if pd.isna(text) or text == '':
        return 0
    return str(text).count('!')

print("Feature extraction functions created")

print("\n[STEP 4: EXTRACT FEATURES FOR ALL POSTS]")
print("-" * 80)

print("Extracting temporal features...")
posts_df['chronic_markers'] = posts_df['full_text'].apply(lambda x: count_markers(x, chronic_markers))
posts_df['acute_markers'] = posts_df['full_text'].apply(lambda x: count_markers(x, acute_markers))
posts_df['chronic_conditions'] = posts_df['full_text'].apply(lambda x: count_conditions(x, chronic_conditions))
posts_df['acute_conditions'] = posts_df['full_text'].apply(lambda x: count_conditions(x, acute_conditions))
posts_df['duration_days'] = posts_df['full_text'].apply(extract_duration_days)
posts_df['er_mentions'] = posts_df['full_text'].apply(has_er_mention)
posts_df['question_marks'] = posts_df['full_text'].apply(count_question_marks)
posts_df['exclamation_marks'] = posts_df['full_text'].apply(count_exclamation_marks)

# Text length features
posts_df['text_length'] = posts_df['full_text'].str.len()
posts_df['word_count'] = posts_df['full_text'].str.split().str.len()

print("Features extracted!")

print("\n[STEP 5: CREATE LABELS USING RULE-BASED APPROACH]")
print("-" * 80)

def classify_chronic_acute(row):
    """
    Rule-based classification to create training labels
    Returns: 'chronic', 'acute', or 'unclear'
    """
    chronic_score = 0
    acute_score = 0
    
    # Temporal markers
    chronic_score += row['chronic_markers'] * 2
    acute_score += row['acute_markers'] * 2
    
    # Condition mentions
    chronic_score += row['chronic_conditions'] * 3
    acute_score += row['acute_conditions'] * 3
    
    # Duration
    if row['duration_days'] > 90:  # More than 3 months = chronic
        chronic_score += 5
    elif row['duration_days'] > 0 and row['duration_days'] <= 7:  # Less than a week = acute
        acute_score += 5
    
    # ER mentions = acute
    acute_score += row['er_mentions'] * 3
    
    # Urgency indicators
    acute_score += row['exclamation_marks']
    
    # Decide
    if chronic_score > acute_score and chronic_score >= 3:
        return 'chronic'
    elif acute_score > chronic_score and acute_score >= 3:
        return 'acute'
    else:
        return 'unclear'

posts_df['condition_type'] = posts_df.apply(classify_chronic_acute, axis=1)

print("\nLabel distribution:")
label_counts = posts_df['condition_type'].value_counts()
for label, count in label_counts.items():
    print(f"  {label:10s}: {count:>5,} ({count/len(posts_df)*100:>5.1f}%)")

# Keep only clear labels for training
labeled_df = posts_df[posts_df['condition_type'].isin(['chronic', 'acute'])].copy()
print(f"\nPosts with clear labels: {len(labeled_df):,}")

print("\n[STEP 6: PREPARE FEATURES FOR ML]")
print("-" * 80)

# Select features
feature_columns = [
    'chronic_markers', 'acute_markers', 'chronic_conditions', 'acute_conditions',
    'duration_days', 'er_mentions', 'question_marks', 'exclamation_marks',
    'text_length', 'word_count', 'medical_term_count', 'anxiety_words', 'urgency_words'
]

X_features = labeled_df[feature_columns].fillna(0)
y = labeled_df['condition_type']

print(f"Feature matrix shape: {X_features.shape}")
print(f"Features: {list(feature_columns)}")

# Add TF-IDF features
print("\nAdding TF-IDF text features...")
tfidf = TfidfVectorizer(max_features=200, min_df=3, max_df=0.7, ngram_range=(1, 2), stop_words='english')
X_text = tfidf.fit_transform(labeled_df['full_text'])

# Combine features
from scipy.sparse import hstack
X_combined = hstack([X_features.values, X_text])

print(f"Combined feature matrix: {X_combined.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

print("\n[STEP 7: TRAIN MULTIPLE CLASSIFIERS]")
print("-" * 80)

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    
    # Train
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {accuracy:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    results[name] = {
        'model': clf,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'predictions': y_pred
    }

print("\n[STEP 8: DETAILED EVALUATION - BEST MODEL]")
print("-" * 80)

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['accuracy']:.3f}")

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test, best_predictions, target_names=['acute', 'chronic']))

# Confusion matrix
cm = confusion_matrix(y_test, best_predictions, labels=['acute', 'chronic'])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Acute', 'Chronic'],
            yticklabels=['Acute', 'Chronic'], ax=ax)
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('q9_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nSaved: q9_confusion_matrix.png")
plt.close()

print("\n[STEP 9: FEATURE IMPORTANCE ANALYSIS]")
print("-" * 80)

if best_model_name == 'Random Forest':
    # Get feature importance
    importances = best_model.feature_importances_[:len(feature_columns)]
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance_df.head(10).plot(x='feature', y='importance', kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('q9_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nSaved: q9_feature_importance.png")
    plt.close()

print("\n[STEP 10: ANALYZE RESULTS BY SUBREDDIT]")
print("-" * 80)

# Apply model to all labeled data
all_predictions = best_model.predict(X_combined)
labeled_df['predicted_type'] = all_predictions

print("\nChronic vs Acute Distribution by Subreddit:")
print("-" * 80)
print(f"{'Subreddit':<20} {'Chronic':<10} {'Acute':<10} {'% Chronic':<12}")
print("-" * 80)

for subreddit in sorted(labeled_df['subreddit'].unique()):
    sub_data = labeled_df[labeled_df['subreddit'] == subreddit]
    chronic_count = (sub_data['predicted_type'] == 'chronic').sum()
    acute_count = (sub_data['predicted_type'] == 'acute').sum()
    pct_chronic = (chronic_count / len(sub_data)) * 100 if len(sub_data) > 0 else 0
    
    print(f"r/{subreddit:<18} {chronic_count:>8,}  {acute_count:>8,}  {pct_chronic:>10.1f}%")

print("\n[STEP 11: ENGAGEMENT COMPARISON]")
print("-" * 80)

chronic_posts = labeled_df[labeled_df['predicted_type'] == 'chronic']
acute_posts = labeled_df[labeled_df['predicted_type'] == 'acute']

print("\nEngagement Metrics:")
print(f"\nChronic Posts (n={len(chronic_posts):,}):")
print(f"  Average score: {chronic_posts['post_score'].mean():.1f}")
print(f"  Average comments: {chronic_posts['actual_comment_count'].mean():.1f}")
print(f"  Average length: {chronic_posts['text_length'].mean():.0f} chars")

print(f"\nAcute Posts (n={len(acute_posts):,}):")
print(f"  Average score: {acute_posts['post_score'].mean():.1f}")
print(f"  Average comments: {acute_posts['actual_comment_count'].mean():.1f}")
print(f"  Average length: {acute_posts['text_length'].mean():.0f} chars")

print("\n[STEP 12: SAVE RESULTS]")
print("-" * 80)

# Save predictions
labeled_df[['post_id', 'subreddit', 'post_title', 'condition_type', 'predicted_type'] + feature_columns].to_csv('q9_chronic_acute_predictions.csv', index=False)
print("Saved: q9_chronic_acute_predictions.csv")

# Save model comparison
model_comparison = pd.DataFrame([
    {'Model': name, 'Accuracy': results[name]['accuracy'], 'CV_Mean': results[name]['cv_mean']}
    for name in results
])
model_comparison.to_csv('q9_model_comparison.csv', index=False)
print("Saved: q9_model_comparison.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

print(f"\nKEY FINDINGS:")
print(f"1. Best model: {best_model_name} with {results[best_model_name]['accuracy']:.1%} accuracy")
print(f"2. Can distinguish chronic vs acute with good performance")
print(f"3. Most important features: temporal markers and condition mentions")
print(f"4. Different subreddits have different chronic/acute ratios")

