"""
Q8: SUBREDDIT DIFFERENCES - VISUALIZATIONS
================================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("=" * 80)
print("Q8: CREATING VISUALIZATIONS")
print("=" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load data
profiles_df = pd.read_csv('q8_subreddit_profiles.csv')
engagement_df = pd.read_csv('q8_engagement_stats.csv')

# Add self-diagnosis data
self_dx_data = {
    'AskDocs': 63.3,
    'Health': 1.9,
    'HealthAnxiety': 33.9,
    'biohackers': 23.2,
    'medicaladvice': 26.6,
    'medicine': 15.5
}
profiles_df['self_diagnosis_pct'] = profiles_df['subreddit'].map(self_dx_data)

print("\n[VIZ 1: Engagement Metrics Comparison]")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Sort by avg_score for better visualization
profiles_sorted = profiles_df.sort_values('avg_score', ascending=True)

# 1. Average Score
axes[0, 0].barh(profiles_sorted['subreddit'], profiles_sorted['avg_score'], color='steelblue')
axes[0, 0].set_xlabel('Average Post Score', fontsize=12)
axes[0, 0].set_title('Average Post Score by Subreddit', fontsize=14, fontweight='bold')
for i, v in enumerate(profiles_sorted['avg_score']):
    axes[0, 0].text(v + 20, i, f'{v:.0f}', va='center')

# 2. Average Comments
profiles_sorted_comments = profiles_df.sort_values('avg_comments', ascending=True)
axes[0, 1].barh(profiles_sorted_comments['subreddit'], profiles_sorted_comments['avg_comments'], color='coral')
axes[0, 1].set_xlabel('Average Comments per Post', fontsize=12)
axes[0, 1].set_title('Average Comments by Subreddit', fontsize=14, fontweight='bold')
for i, v in enumerate(profiles_sorted_comments['avg_comments']):
    axes[0, 1].text(v + 2, i, f'{v:.1f}', va='center')

# 3. Post Length
profiles_sorted_length = profiles_df.sort_values('avg_post_length', ascending=True)
axes[1, 0].barh(profiles_sorted_length['subreddit'], profiles_sorted_length['avg_post_length'], color='mediumseagreen')
axes[1, 0].set_xlabel('Average Post Length (characters)', fontsize=12)
axes[1, 0].set_title('Average Post Length by Subreddit', fontsize=14, fontweight='bold')
for i, v in enumerate(profiles_sorted_length['avg_post_length']):
    axes[1, 0].text(v + 20, i, f'{v:.0f}', va='center')

# 4. Questions Percentage
profiles_sorted_q = profiles_df.sort_values('pct_questions', ascending=True)
axes[1, 1].barh(profiles_sorted_q['subreddit'], profiles_sorted_q['pct_questions'], color='mediumpurple')
axes[1, 1].set_xlabel('Percentage of Posts with Questions (%)', fontsize=12)
axes[1, 1].set_title('Question Posts by Subreddit', fontsize=14, fontweight='bold')
for i, v in enumerate(profiles_sorted_q['pct_questions']):
    axes[1, 1].text(v + 1, i, f'{v:.1f}%', va='center')

plt.tight_layout()
plt.savefig('q8_viz1_engagement_metrics.png', dpi=300, bbox_inches='tight')
print("Saved: q8_viz1_engagement_metrics.png")
plt.close()

# ============================================================================
# VIZ 2: Language Characteristics Heatmap
# ============================================================================
print("\n[VIZ 2: Language Characteristics Heatmap]")

# Prepare data for heatmap
heatmap_data = profiles_df[['subreddit', 'medical_density', 'anxiety_score', 'urgency_score', 'self_diagnosis_pct']].copy()
heatmap_data = heatmap_data.set_index('subreddit')

# Normalize to 0-100 scale for better visualization
heatmap_normalized = heatmap_data.copy()
for col in heatmap_normalized.columns:
    max_val = heatmap_normalized[col].max()
    if max_val > 0:
        heatmap_normalized[col] = (heatmap_normalized[col] / max_val) * 100

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_normalized.T, annot=heatmap_data.T, fmt='.2f', cmap='YlOrRd', 
            cbar_kws={'label': 'Normalized Score (0-100)'}, ax=ax, linewidths=1)
ax.set_title('Language & Behavior Characteristics by Subreddit', fontsize=14, fontweight='bold')
ax.set_xlabel('Subreddit', fontsize=12)
ax.set_ylabel('Characteristic', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('q8_viz2_language_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: q8_viz2_language_heatmap.png")
plt.close()

# ============================================================================
# VIZ 3: Self-Diagnosis Prevalence
# ============================================================================
print("\n[VIZ 3: Self-Diagnosis Prevalence]")

fig, ax = plt.subplots(figsize=(12, 7))
profiles_sorted_dx = profiles_df.sort_values('self_diagnosis_pct', ascending=True)

colors = ['#d62728' if x > 50 else '#ff7f0e' if x > 25 else '#2ca02c' for x in profiles_sorted_dx['self_diagnosis_pct']]
bars = ax.barh(profiles_sorted_dx['subreddit'], profiles_sorted_dx['self_diagnosis_pct'], color=colors)

ax.set_xlabel('Percentage of Posts with Self-Diagnosis (%)', fontsize=12)
ax.set_title('Self-Diagnosis Prevalence by Subreddit', fontsize=14, fontweight='bold')
ax.axvline(x=25, color='gray', linestyle='--', alpha=0.5, label='25% threshold')
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')

for i, v in enumerate(profiles_sorted_dx['self_diagnosis_pct']):
    ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')

ax.legend()
plt.tight_layout()
plt.savefig('q8_viz3_self_diagnosis.png', dpi=300, bbox_inches='tight')
print("Saved: q8_viz3_self_diagnosis.png")
plt.close()

# ============================================================================
# VIZ 4: Emotional Language Comparison
# ============================================================================
print("\n[VIZ 4: Emotional Language Comparison]")

fig, ax = plt.subplots(figsize=(12, 7))

subreddits = profiles_df['subreddit'].tolist()
x = np.arange(len(subreddits))
width = 0.25

anxiety_bars = ax.bar(x - width, profiles_df['anxiety_score'], width, label='Anxiety', color='#e74c3c')
urgency_bars = ax.bar(x, profiles_df['urgency_score'], width, label='Urgency', color='#f39c12')
uncertainty_bars = ax.bar(x + width, profiles_df['pct_questions']/20, width, label='Questions (scaled)', color='#3498db')

ax.set_xlabel('Subreddit', fontsize=12)
ax.set_ylabel('Average Word Count per Post', fontsize=12)
ax.set_title('Emotional Language Patterns by Subreddit', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(subreddits, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('q8_viz4_emotional_language.png', dpi=300, bbox_inches='tight')
print("Saved: q8_viz4_emotional_language.png")
plt.close()

# ============================================================================
# VIZ 5: Radar Chart - Subreddit Profiles
# ============================================================================
print("\n[VIZ 5: Radar Chart - Subreddit Personality Profiles]")

from math import pi

# Select key metrics and normalize
metrics = ['avg_score', 'avg_comments', 'medical_density', 'anxiety_score', 'self_diagnosis_pct']
metric_labels = ['Engagement\n(Score)', 'Discussion\n(Comments)', 'Medical\nTerminology', 'Anxiety\nLevel', 'Self-Diagnosis\nRate']

# Normalize each metric to 0-1 scale
normalized_df = profiles_df[['subreddit'] + metrics].copy()
for metric in metrics:
    max_val = normalized_df[metric].max()
    if max_val > 0:
        normalized_df[metric] = normalized_df[metric] / max_val

# Create radar chart for each subreddit
fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

for idx, (_, row) in enumerate(normalized_df.iterrows()):
    ax = axes[idx]
    
    values = row[metrics].tolist()
    values += values[:1]  # Complete the circle
    
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=row['subreddit'])
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=8)
    ax.set_ylim(0, 1)
    ax.set_title(f"r/{row['subreddit']}", size=14, fontweight='bold', pad=20)
    ax.grid(True)

plt.tight_layout()
plt.savefig('q8_viz5_radar_profiles.png', dpi=300, bbox_inches='tight')
print("Saved: q8_viz5_radar_profiles.png")
plt.close()

# ============================================================================
# VIZ 6: Medical Terminology vs Engagement
# ============================================================================
print("\n[VIZ 6: Medical Terminology vs Engagement Scatter]")

fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(profiles_df['medical_density'], profiles_df['avg_score'], 
                     s=profiles_df['avg_comments']*3, alpha=0.6, 
                     c=profiles_df['self_diagnosis_pct'], cmap='RdYlGn_r')

# Add labels
for idx, row in profiles_df.iterrows():
    ax.annotate(f"r/{row['subreddit']}", 
                (row['medical_density'], row['avg_score']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

ax.set_xlabel('Medical Terminology Density (%)', fontsize=12)
ax.set_ylabel('Average Post Score', fontsize=12)
ax.set_title('Medical Terminology vs Engagement\n(bubble size = avg comments)', 
             fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Self-Diagnosis Rate (%)', fontsize=10)

plt.tight_layout()
plt.savefig('q8_viz6_terminology_engagement.png', dpi=300, bbox_inches='tight')
print("Saved: q8_viz6_terminology_engagement.png")
plt.close()

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS CREATED!")
print("=" * 80)
print("\nGenerated files:")
print("  1. q8_viz1_engagement_metrics.png")
print("  2. q8_viz2_language_heatmap.png")
print("  3. q8_viz3_self_diagnosis.png")
print("  4. q8_viz4_emotional_language.png")
print("  5. q8_viz5_radar_profiles.png")
print("  6. q8_viz6_terminology_engagement.png")

