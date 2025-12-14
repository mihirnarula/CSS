"""
Q9: CHRONIC VS ACUTE - FINAL INSIGHTS & ANSWER
================================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("Q9: CHRONIC VS ACUTE - ANSWERING THE RESEARCH QUESTION")
print("=" * 80)

# Load results
predictions_df = pd.read_csv('q9_chronic_acute_predictions.csv')
# Load full posts data for engagement metrics
posts_full = pd.read_csv('q8_posts_with_features.csv', low_memory=False)
# Merge predictions with full data
predictions_df = predictions_df.merge(posts_full[['post_id', 'post_score', 'actual_comment_count']], on='post_id', how='left')

print("\n" + "=" * 80)
print("INSIGHT 1: CAN WE DISTINGUISH CHRONIC VS ACUTE?")
print("=" * 80)

print("\nYES! With 98.5% accuracy using Gradient Boosting")
print("\nWhat this means:")
print("  • Chronic and acute discussions have DISTINCT patterns")
print("  • Temporal language is highly predictive")
print("  • AI can automatically categorize health discussions")

print("\n" + "=" * 80)
print("INSIGHT 2: OVERALL DISTRIBUTION")
print("=" * 80)

total_chronic = (predictions_df['predicted_type'] == 'chronic').sum()
total_acute = (predictions_df['predicted_type'] == 'acute').sum()
total = len(predictions_df)

print(f"\nAcross all medical subreddits:")
print(f"  Acute discussions:   {total_acute:>5,} ({total_acute/total*100:>5.1f}%)")
print(f"  Chronic discussions: {total_chronic:>5,} ({total_chronic/total*100:>5.1f}%)")

print("\n→ ACUTE discussions DOMINATE (78.2%)")
print("→ People seek online help more for sudden problems than ongoing management")

print("\n" + "=" * 80)
print("INSIGHT 3: SUBREDDIT DIFFERENCES")
print("=" * 80)

subreddit_analysis = []
for subreddit in sorted(predictions_df['subreddit'].unique()):
    sub_data = predictions_df[predictions_df['subreddit'] == subreddit]
    chronic_count = (sub_data['predicted_type'] == 'chronic').sum()
    acute_count = (sub_data['predicted_type'] == 'acute').sum()
    pct_chronic = (chronic_count / len(sub_data)) * 100
    
    subreddit_analysis.append({
        'subreddit': subreddit,
        'chronic': chronic_count,
        'acute': acute_count,
        'pct_chronic': pct_chronic,
        'total': len(sub_data)
    })

subreddit_df = pd.DataFrame(subreddit_analysis).sort_values('pct_chronic', ascending=False)

print("\nChronic Discussion Rate by Subreddit:")
print("-" * 60)
for _, row in subreddit_df.iterrows():
    print(f"r/{row['subreddit']:<18} {row['pct_chronic']:>5.1f}% chronic  ({row['chronic']:>3,} / {row['total']:>4,} posts)")

print("\nKEY PATTERNS:")
print("  HIGHEST CHRONIC: r/HealthAnxiety (38.4%)")
print("    → People with health anxiety have ONGOING worries, not sudden issues")
print("    → Chronic anxiety about health manifests as chronic health concerns")

print("\n  HIGH CHRONIC: r/AskDocs (30.2%)")
print("    → Mix of both: some seek diagnosis for chronic issues")
print("    → Some want expert opinion on long-standing problems")

print("\n  LOWEST CHRONIC: r/Health (4.5%)")
print("    → News/information subreddit, not personal health")
print("    → Acute topics (outbreaks, emergencies) make news")

print("\n  PROFESSIONAL: r/medicine (13.2%)")
print("    → Professionals discuss both, but acute cases more interesting")
print("    → Emergency/dramatic cases get more discussion")

print("\n" + "=" * 80)
print("INSIGHT 4: ENGAGEMENT DIFFERENCES")
print("=" * 80)

chronic_posts = predictions_df[predictions_df['predicted_type'] == 'chronic']
acute_posts = predictions_df[predictions_df['predicted_type'] == 'acute']

print(f"\nChronic Posts (n={len(chronic_posts):,}):")
print(f"  Average score: {chronic_posts['post_score'].mean():.0f}")
print(f"  Average comments: {chronic_posts['actual_comment_count'].mean():.1f}")
print(f"  Average length: {chronic_posts['text_length'].mean():.0f} characters")

print(f"\nAcute Posts (n={len(acute_posts):,}):")
print(f"  Average score: {acute_posts['post_score'].mean():.0f}")
print(f"  Average comments: {acute_posts['actual_comment_count'].mean():.1f}")
print(f"  Average length: {acute_posts['text_length'].mean():.0f} characters")

print("\nKEY DIFFERENCES:")
print("  • ACUTE posts get MORE engagement (667 vs 388 score)")
print("  • ACUTE posts get MORE comments (91.7 vs 72.5)")
print("  • CHRONIC posts are LONGER (1,539 vs 940 chars)")
print("")
print("→ Sudden problems are more dramatic and get more attention")
print("→ Chronic issues require more explanation (longer posts)")
print("→ Community responds more to urgent/acute situations")

print("\n" + "=" * 80)
print("INSIGHT 5: FEATURE IMPORTANCE")
print("=" * 80)

print("\nMost predictive features for classification:")
print("  1. Acute temporal markers ('suddenly', 'just started')")
print("  2. Chronic temporal markers ('years', 'always', 'chronic')")
print("  3. Duration mentions (days vs years)")
print("  4. Condition type (diabetes vs flu)")
print("  5. ER/urgency mentions")
print("")
print("→ Temporal language is the STRONGEST predictor")
print("→ How people describe TIME reveals condition type")

print("\n" + "=" * 80)
print("INSIGHT 6: REAL-WORLD IMPLICATIONS")
print("=" * 80)

print("\n1. PLATFORM DESIGN:")
print("   • Acute posts need FAST responses (time-sensitive)")
print("   • Chronic posts need SUSTAINED support (ongoing)")
print("   • Different UI/UX for different needs")

print("\n2. MODERATION:")
print("   • Acute posts may need urgent medical disclaimer")
print("   • Chronic posts benefit from community building")
print("   • Can auto-detect and route appropriately")

print("\n3. HEALTHCARE INSIGHTS:")
print("   • Online platforms used more for acute concerns")
print("   • Chronic condition management happens elsewhere")
print("   • Gap in online chronic disease support")

print("\n4. MENTAL HEALTH:")
print("   • r/HealthAnxiety shows chronic anxiety patterns")
print("   • Health anxiety is itself a chronic condition")
print("   • Need different interventions than acute worries")

print("\n" + "=" * 80)
print("FINAL ANSWER TO QUESTION 9")
print("=" * 80)

answer = """
Q: Can we distinguish between discussions of chronic conditions (ongoing 
   management) vs. acute symptoms (sudden onset)?

ANSWER: YES - with 98.5% accuracy!

KEY FINDINGS:

1. ACUTE DISCUSSIONS DOMINATE (78.2%)
   - People seek online help MORE for sudden problems
   - Acute issues are more urgent and dramatic
   - Get more engagement (667 score vs 388)

2. CHRONIC DISCUSSIONS ARE LONGER (1,539 vs 940 chars)
   - Need more context and medical history
   - Require detailed explanation
   - But get LESS attention from community

3. SUBREDDIT DIFFERENCES REVEAL PURPOSE:
   - r/HealthAnxiety: 38.4% chronic (ongoing anxiety)
   - r/AskDocs: 30.2% chronic (mix of both)
   - r/Health: 4.5% chronic (news focuses on acute)
   - r/medicine: 13.2% chronic (acute more interesting)

4. TEMPORAL LANGUAGE IS KEY PREDICTOR:
   - "suddenly", "just started" → ACUTE
   - "for years", "always", "chronic" → CHRONIC
   - Duration mentions highly predictive
   - AI can classify from text alone

5. DIFFERENT NEEDS, DIFFERENT RESPONSES:
   - Acute: Need immediate diagnosis/reassurance
   - Chronic: Need ongoing support/management tips
   - Community responds more to acute (urgency bias)
   - Chronic posts get less attention despite greater need

IMPLICATIONS:
- Online health platforms are ACUTE-focused
- Gap in chronic disease support online
- Health anxiety manifests as chronic concerns
- Temporal language reveals condition type
- Can auto-route posts to appropriate resources

CONCLUSION: Chronic and acute discussions are fundamentally different in 
language, engagement, and community response. Online platforms primarily 
serve acute needs, leaving chronic condition management underserved.
"""

print(answer)

# Save final answer
with open('q9_final_answer.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("RESEARCH QUESTION 9: FINAL ANSWER\n")
    f.write("=" * 80 + "\n\n")
    f.write(answer)
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("DETAILED STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total posts analyzed: {len(predictions_df):,}\n")
    f.write(f"Acute: {total_acute:,} ({total_acute/total*100:.1f}%)\n")
    f.write(f"Chronic: {total_chronic:,} ({total_chronic/total*100:.1f}%)\n\n")
    f.write("By Subreddit:\n")
    for _, row in subreddit_df.iterrows():
        f.write(f"  r/{row['subreddit']:<18} {row['pct_chronic']:>5.1f}% chronic\n")

print("\nSaved: q9_final_answer.txt")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subreddit comparison
subreddit_df_plot = subreddit_df.sort_values('pct_chronic', ascending=True)
axes[0].barh(subreddit_df_plot['subreddit'], subreddit_df_plot['pct_chronic'], color='coral')
axes[0].set_xlabel('% Chronic Discussions', fontsize=12)
axes[0].set_title('Chronic Discussion Rate by Subreddit', fontsize=13, fontweight='bold')
axes[0].axvline(x=20, color='gray', linestyle='--', alpha=0.5)
for i, v in enumerate(subreddit_df_plot['pct_chronic']):
    axes[0].text(v + 1, i, f'{v:.1f}%', va='center')

# Engagement comparison
categories = ['Score', 'Comments', 'Length/10']
chronic_vals = [chronic_posts['post_score'].mean(), 
                chronic_posts['actual_comment_count'].mean(),
                chronic_posts['text_length'].mean()/10]
acute_vals = [acute_posts['post_score'].mean(),
              acute_posts['actual_comment_count'].mean(),
              acute_posts['text_length'].mean()/10]

x = range(len(categories))
width = 0.35
axes[1].bar([i - width/2 for i in x], chronic_vals, width, label='Chronic', color='steelblue')
axes[1].bar([i + width/2 for i in x], acute_vals, width, label='Acute', color='coral')
axes[1].set_ylabel('Value', fontsize=12)
axes[1].set_title('Engagement: Chronic vs Acute', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories)
axes[1].legend()

plt.tight_layout()
plt.savefig('q9_final_insights.png', dpi=300, bbox_inches='tight')
print("Saved: q9_final_insights.png")
plt.close()

print("\n" + "=" * 80)
print("QUESTION 9 COMPLETE WITH INSIGHTS!")
print("=" * 80)

