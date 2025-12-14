"""
Q1: SYMPTOM ANALYSIS - VISUALIZATIONS
================================================================================
Creating visual representations of symptom patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast

print("=" * 80)
print("Q1: CREATING VISUALIZATIONS")
print("=" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("\n[STEP 1: Loading processed data]")
df = pd.read_csv('q1_symptom_features.csv', nrows=50000)
print(f"Loaded {len(df):,} rows")

# Convert string representations back to dicts/lists
print("\n[STEP 2: Parsing extracted features]")
df['post_symptoms'] = df['post_symptoms'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != '{}' else {})
df['post_body_parts'] = df['post_body_parts'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else [])
df['post_conditions'] = df['post_conditions'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else [])

print("\n[STEP 3: Aggregating data for visualizations]")

# Overall symptom counts
all_symptoms = Counter()
for symptoms_dict in df['post_symptoms']:
    for category, count in symptoms_dict.items():
        all_symptoms[category] += count

# Subreddit-specific symptoms
subreddit_symptoms = {}
for subreddit in df['subreddit'].unique():
    sub_df = df[df['subreddit'] == subreddit]
    sub_symptoms = Counter()
    for symptoms_dict in sub_df['post_symptoms']:
        for category, count in symptoms_dict.items():
            sub_symptoms[category] += count
    subreddit_symptoms[subreddit] = sub_symptoms

# Body parts
all_body_parts = Counter()
for parts_list in df['post_body_parts']:
    for part in parts_list:
        all_body_parts[part] += 1

# Conditions
all_conditions = Counter()
for conditions_list in df['post_conditions']:
    for condition in conditions_list:
        all_conditions[condition] += 1

print("Aggregation complete!")

# ============================================================================
# VISUALIZATION 1: Overall Top Symptoms
# ============================================================================
print("\n[VIZ 1: Bar Chart - Top 10 Symptom Categories]")

fig, ax = plt.subplots(figsize=(12, 6))
top_symptoms = all_symptoms.most_common(10)
categories = [cat for cat, _ in top_symptoms]
counts = [count for _, count in top_symptoms]

bars = ax.barh(categories, counts, color='steelblue')
ax.set_xlabel('Number of Mentions', fontsize=12)
ax.set_ylabel('Symptom Category', fontsize=12)
ax.set_title('Top 10 Symptom Categories Across All Subreddits', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add value labels
for i, (bar, count) in enumerate(zip(bars, counts)):
    ax.text(count + 100, i, f'{count:,}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('q1_viz1_top_symptoms.png', dpi=300, bbox_inches='tight')
print("Saved: q1_viz1_top_symptoms.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Symptoms by Subreddit Heatmap
# ============================================================================
print("\n[VIZ 2: Heatmap - Symptoms by Subreddit]")

# Create matrix for heatmap
symptom_categories = list(all_symptoms.keys())
subreddits = list(subreddit_symptoms.keys())

# Build matrix
matrix = []
for subreddit in subreddits:
    row = []
    for symptom in symptom_categories:
        count = subreddit_symptoms[subreddit].get(symptom, 0)
        row.append(count)
    matrix.append(row)

# Normalize by subreddit (percentage of total mentions in that subreddit)
matrix_normalized = []
for row in matrix:
    total = sum(row)
    if total > 0:
        normalized_row = [(val / total) * 100 for val in row]
    else:
        normalized_row = row
    matrix_normalized.append(normalized_row)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(matrix_normalized, 
            xticklabels=symptom_categories, 
            yticklabels=subreddits,
            cmap='YlOrRd', 
            annot=True, 
            fmt='.1f',
            cbar_kws={'label': 'Percentage of Mentions (%)'},
            ax=ax)

ax.set_title('Symptom Category Distribution by Subreddit (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Symptom Category', fontsize=12)
ax.set_ylabel('Subreddit', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('q1_viz2_subreddit_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: q1_viz2_subreddit_heatmap.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Top Body Parts
# ============================================================================
print("\n[VIZ 3: Bar Chart - Top 15 Body Parts Mentioned]")

fig, ax = plt.subplots(figsize=(12, 7))
top_parts = all_body_parts.most_common(15)
parts = [part for part, _ in top_parts]
counts = [count for _, count in top_parts]

bars = ax.barh(parts, counts, color='coral')
ax.set_xlabel('Number of Mentions', fontsize=12)
ax.set_ylabel('Body Part', fontsize=12)
ax.set_title('Top 15 Body Parts Mentioned in Posts', fontsize=14, fontweight='bold')
ax.invert_yaxis()

for i, (bar, count) in enumerate(zip(bars, counts)):
    ax.text(count + 50, i, f'{count:,}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('q1_viz3_body_parts.png', dpi=300, bbox_inches='tight')
print("Saved: q1_viz3_body_parts.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: Top Medical Conditions
# ============================================================================
print("\n[VIZ 4: Bar Chart - Top 15 Medical Conditions]")

fig, ax = plt.subplots(figsize=(12, 7))
top_conditions = all_conditions.most_common(15)
conditions = [cond for cond, _ in top_conditions]
counts = [count for _, count in top_conditions]

bars = ax.barh(conditions, counts, color='mediumseagreen')
ax.set_xlabel('Number of Mentions', fontsize=12)
ax.set_ylabel('Medical Condition', fontsize=12)
ax.set_title('Top 15 Medical Conditions Mentioned in Posts', fontsize=14, fontweight='bold')
ax.invert_yaxis()

for i, (bar, count) in enumerate(zip(bars, counts)):
    ax.text(count + 50, i, f'{count:,}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('q1_viz4_conditions.png', dpi=300, bbox_inches='tight')
print("Saved: q1_viz4_conditions.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: Symptom Co-occurrence Analysis
# ============================================================================
print("\n[VIZ 5: Symptom Co-occurrence Matrix]")

# Build co-occurrence matrix
symptom_cats = list(all_symptoms.keys())
cooccurrence = {cat: Counter() for cat in symptom_cats}

for symptoms_dict in df['post_symptoms']:
    if len(symptoms_dict) > 1:
        cats = list(symptoms_dict.keys())
        for i, cat1 in enumerate(cats):
            for cat2 in cats[i+1:]:
                cooccurrence[cat1][cat2] += 1
                cooccurrence[cat2][cat1] += 1

# Create matrix
co_matrix = []
for cat1 in symptom_cats:
    row = []
    for cat2 in symptom_cats:
        if cat1 == cat2:
            row.append(0)
        else:
            row.append(cooccurrence[cat1].get(cat2, 0))
    co_matrix.append(row)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(co_matrix, 
            xticklabels=symptom_cats, 
            yticklabels=symptom_cats,
            cmap='Blues', 
            annot=True, 
            fmt='d',
            cbar_kws={'label': 'Co-occurrence Count'},
            ax=ax)

ax.set_title('Symptom Category Co-occurrence in Same Posts', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('q1_viz5_cooccurrence.png', dpi=300, bbox_inches='tight')
print("Saved: q1_viz5_cooccurrence.png")
plt.close()

# ============================================================================
# VISUALIZATION 6: Engagement Analysis
# ============================================================================
print("\n[VIZ 6: Symptom Mentions vs Post Engagement]")

# Count symptoms per post
df['symptom_count'] = df['post_symptoms'].apply(lambda x: sum(x.values()) if isinstance(x, dict) else 0)
df['has_symptoms'] = df['symptom_count'] > 0

# Compare engagement
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Post score comparison
with_symptoms = df[df['has_symptoms']]['post_score']
without_symptoms = df[~df['has_symptoms']]['post_score']

axes[0].boxplot([with_symptoms, without_symptoms], labels=['With Symptoms', 'Without Symptoms'])
axes[0].set_ylabel('Post Score', fontsize=12)
axes[0].set_title('Post Scores: With vs Without Symptom Mentions', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 100)  # Focus on lower range
axes[0].grid(axis='y', alpha=0.3)

# Comment count comparison
with_symptoms_comments = df[df['has_symptoms']]['post_num_comments']
without_symptoms_comments = df[~df['has_symptoms']]['post_num_comments']

axes[1].boxplot([with_symptoms_comments, without_symptoms_comments], 
                labels=['With Symptoms', 'Without Symptoms'])
axes[1].set_ylabel('Number of Comments', fontsize=12)
axes[1].set_title('Comment Counts: With vs Without Symptom Mentions', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 100)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('q1_viz6_engagement.png', dpi=300, bbox_inches='tight')
print("Saved: q1_viz6_engagement.png")
plt.close()

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS CREATED!")
print("=" * 80)
print("\nGenerated files:")
print("  1. q1_viz1_top_symptoms.png - Top symptom categories")
print("  2. q1_viz2_subreddit_heatmap.png - Symptoms by subreddit")
print("  3. q1_viz3_body_parts.png - Most mentioned body parts")
print("  4. q1_viz4_conditions.png - Most mentioned conditions")
print("  5. q1_viz5_cooccurrence.png - Symptom co-occurrence patterns")
print("  6. q1_viz6_engagement.png - Engagement analysis")

