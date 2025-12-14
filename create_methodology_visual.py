"""
Create visual summary of methodology
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'PROJECT METHODOLOGY OVERVIEW', 
        ha='center', va='top', fontsize=20, fontweight='bold')
ax.text(5, 11, '"The Internet as a Doctor: How People Self-diagnose Online"',
        ha='center', va='top', fontsize=14, style='italic')

# Color scheme
colors = {
    'data': '#3498db',
    'feature': '#e74c3c',
    'nlp': '#2ecc71',
    'ml': '#f39c12',
    'analysis': '#9b59b6'
}

y_pos = 10

# 1. DATA COLLECTION
box1 = FancyBboxPatch((0.2, y_pos-0.8), 4.6, 0.7, 
                       boxstyle="round,pad=0.1", 
                       edgecolor=colors['data'], 
                       facecolor=colors['data'], 
                       alpha=0.3, linewidth=2)
ax.add_patch(box1)
ax.text(2.5, y_pos-0.45, '1. DATA COLLECTION & CLEANING', 
        ha='center', fontsize=12, fontweight='bold')
ax.text(0.5, y_pos-1.2, '• 312,342 comments', fontsize=9)
ax.text(0.5, y_pos-1.5, '• 3,573 posts', fontsize=9)
ax.text(0.5, y_pos-1.8, '• 6 subreddits', fontsize=9)
ax.text(2.8, y_pos-1.2, '• 95.16% data retained', fontsize=9)
ax.text(2.8, y_pos-1.5, '• 14+ years (2011-2025)', fontsize=9)
ax.text(2.8, y_pos-1.8, '• Removed deleted content', fontsize=9)

y_pos -= 2.5

# 2. FEATURE ENGINEERING
box2 = FancyBboxPatch((5.2, y_pos-0.8), 4.6, 0.7,
                       boxstyle="round,pad=0.1",
                       edgecolor=colors['feature'],
                       facecolor=colors['feature'],
                       alpha=0.3, linewidth=2)
ax.add_patch(box2)
ax.text(7.5, y_pos-0.45, '2. FEATURE ENGINEERING', 
        ha='center', fontsize=12, fontweight='bold')
ax.text(5.5, y_pos-1.2, '• 50+ features created', fontsize=9)
ax.text(5.5, y_pos-1.5, '• Text: length, complexity', fontsize=9)
ax.text(5.5, y_pos-1.8, '• Temporal: chronic/acute', fontsize=9)
ax.text(7.8, y_pos-1.2, '• Emotional: anxiety, urgency', fontsize=9)
ax.text(7.8, y_pos-1.5, '• Behavioral: self-diagnosis', fontsize=9)
ax.text(7.8, y_pos-1.8, '• Medical: symptoms, conditions', fontsize=9)

y_pos -= 2.5

# 3. NLP TECHNIQUES
box3 = FancyBboxPatch((0.2, y_pos-0.8), 4.6, 0.7,
                       boxstyle="round,pad=0.1",
                       edgecolor=colors['nlp'],
                       facecolor=colors['nlp'],
                       alpha=0.3, linewidth=2)
ax.add_patch(box3)
ax.text(2.5, y_pos-0.45, '3. NLP TECHNIQUES', 
        ha='center', fontsize=12, fontweight='bold')
ax.text(0.5, y_pos-1.2, '• TF-IDF vectorization', fontsize=9)
ax.text(0.5, y_pos-1.5, '• Sentiment analysis', fontsize=9)
ax.text(0.5, y_pos-1.8, '• Pattern matching (regex)', fontsize=9)
ax.text(2.8, y_pos-1.2, '• Semantic similarity', fontsize=9)
ax.text(2.8, y_pos-1.5, '• Named entity recognition', fontsize=9)
ax.text(2.8, y_pos-1.8, '• Stance detection', fontsize=9)

y_pos -= 2.5

# 4. MACHINE LEARNING
box4 = FancyBboxPatch((5.2, y_pos-0.8), 4.6, 0.7,
                       boxstyle="round,pad=0.1",
                       edgecolor=colors['ml'],
                       facecolor=colors['ml'],
                       alpha=0.3, linewidth=2)
ax.add_patch(box4)
ax.text(7.5, y_pos-0.45, '4. MACHINE LEARNING', 
        ha='center', fontsize=12, fontweight='bold')
ax.text(5.5, y_pos-1.2, '• Logistic Regression', fontsize=9)
ax.text(5.5, y_pos-1.5, '• Random Forest', fontsize=9)
ax.text(5.5, y_pos-1.8, '• Gradient Boosting (best)', fontsize=9)
ax.text(7.8, y_pos-1.2, '• K-Means clustering', fontsize=9)
ax.text(7.8, y_pos-1.5, '• Topic modeling (LDA)', fontsize=9)
ax.text(7.8, y_pos-1.8, '• 5-fold cross-validation', fontsize=9)

y_pos -= 2.5

# 5. STATISTICAL ANALYSIS
box5 = FancyBboxPatch((0.2, y_pos-0.8), 9.6, 0.7,
                       boxstyle="round,pad=0.1",
                       edgecolor=colors['analysis'],
                       facecolor=colors['analysis'],
                       alpha=0.3, linewidth=2)
ax.add_patch(box5)
ax.text(5, y_pos-0.45, '5. STATISTICAL ANALYSIS & VALIDATION', 
        ha='center', fontsize=12, fontweight='bold')
ax.text(0.5, y_pos-1.2, '• ANOVA testing', fontsize=9)
ax.text(0.5, y_pos-1.5, '• Chi-square tests', fontsize=9)
ax.text(2.8, y_pos-1.2, '• Correlation analysis', fontsize=9)
ax.text(2.8, y_pos-1.5, '• Significance testing (p<0.05)', fontsize=9)
ax.text(5.5, y_pos-1.2, '• Confusion matrices', fontsize=9)
ax.text(5.5, y_pos-1.5, '• Feature importance', fontsize=9)
ax.text(7.8, y_pos-1.2, '• 80/20 train-test split', fontsize=9)
ax.text(7.8, y_pos-1.5, '• Reproducibility (seed=42)', fontsize=9)

# Key Results Box
y_pos -= 2.2
result_box = FancyBboxPatch((1, y_pos-1.2), 8, 1.1,
                            boxstyle="round,pad=0.1",
                            edgecolor='black',
                            facecolor='lightyellow',
                            alpha=0.8, linewidth=3)
ax.add_patch(result_box)
ax.text(5, y_pos-0.3, 'KEY RESULTS', ha='center', fontsize=13, fontweight='bold')
ax.text(1.5, y_pos-0.7, '✓ 31.6% posts contain self-diagnosis', fontsize=10)
ax.text(1.5, y_pos-1.0, '✓ 79.8% accuracy classifying subreddits', fontsize=10)
ax.text(5.5, y_pos-0.7, '✓ 98.5% accuracy chronic vs acute', fontsize=10)
ax.text(5.5, y_pos-1.0, '✓ 89% posts have contradictory advice', fontsize=10)

# Tools used
ax.text(5, 0.5, 'Tools: Python • pandas • scikit-learn • NLTK • TextBlob • matplotlib • seaborn',
        ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.savefig('methodology_overview.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: methodology_overview.png")
plt.close()

print("\nMethodology visual created successfully!")

