"""
Q8: SUBREDDIT FUNCTIONAL DIFFERENCES - FINAL INSIGHTS
================================================================================
ANSWERING: How do user behaviors differ across subreddits?
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("=" * 80)
print("Q8: HOW DO SUBREDDITS ACTUALLY DIFFER?")
print("ANSWERING THE RESEARCH QUESTION")
print("=" * 80)

# Load all our analysis
profiles_df = pd.read_csv('q8_subreddit_profiles.csv')
posts_df = pd.read_csv('q8_posts_with_features.csv', low_memory=False)
q2_results = pd.read_csv('q2_posts_analyzed.csv')

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

print("\n" + "=" * 80)
print("INSIGHT 1: WHAT IS EACH SUBREDDIT'S PRIMARY PURPOSE?")
print("=" * 80)

purposes = {
    'AskDocs': {
        'purpose': 'DIAGNOSIS-SEEKING',
        'evidence': [
            f"63.3% of posts contain self-diagnosis attempts (HIGHEST)",
            f"41.7% of posts are questions",
            f"High urgency language (0.93 avg urgency words)",
            f"Longest posts (1,648 chars avg) - detailed symptom descriptions",
            "Top words: 'doctor', 'pain', 'male', 'female', 'thank'"
        ]
    },
    'medicine': {
        'purpose': 'PROFESSIONAL DISCUSSION',
        'evidence': [
            f"Only 15.5% self-diagnosis (professionals discussing cases)",
            f"Highest engagement: 1,359 avg score, 192 avg comments",
            f"Most medical terminology (1.72% density)",
            f"Only 14.8% are questions (mostly statements/discussions)",
            "Top words: 'patient', 'hospital', 'healthcare', 'practice'"
        ]
    },
    'HealthAnxiety': {
        'purpose': 'EMOTIONAL SUPPORT & REASSURANCE',
        'evidence': [
            f"HIGHEST anxiety language (0.86 avg anxiety words)",
            f"33.9% self-diagnosis (seeking validation)",
            f"Most subjective posts (0.503 subjectivity score)",
            f"Medium length posts (972 chars) - emotional narratives",
            "Top words: 'anxiety', 'google', 'symptoms', 'fear', 'reassurance'"
        ]
    },
    'Health': {
        'purpose': 'GENERAL HEALTH NEWS & DISCUSSION',
        'evidence': [
            f"LOWEST self-diagnosis (1.9%) - not personal health issues",
            f"Shortest posts (89 chars) - mostly link titles",
            f"0% have post body (all link posts)",
            f"Only 3.8% are questions",
            "Top words: 'trump', 'fda', 'cdc', 'cancer', 'measles' (news topics)"
        ]
    },
    'biohackers': {
        'purpose': 'OPTIMIZATION & EXPERIMENTATION',
        'evidence': [
            f"23.2% self-diagnosis (experimenting with health)",
            f"36.5% are questions (asking about supplements/effects)",
            f"Most positive sentiment (0.062 polarity)",
            f"Focus on supplements and data",
            "Top words: 'supplements', 'creatine', 'aging', 'effects', 'daily'"
        ]
    },
    'medicaladvice': {
        'purpose': 'QUICK ADVICE-SEEKING',
        'evidence': [
            f"HIGHEST question rate (62.8%)",
            f"LOWEST engagement (2.1 avg score, 1.0 avg comments)",
            f"Short posts (210 chars) - brief questions",
            f"26.6% self-diagnosis",
            "Top words: 'advice', 'pain', 'worried', 'stomach', 'help'"
        ]
    }
}

for subreddit, data in purposes.items():
    print(f"\nr/{subreddit}: {data['purpose']}")
    print("-" * 60)
    for evidence in data['evidence']:
        print(f"  • {evidence}")

print("\n" + "=" * 80)
print("INSIGHT 2: HOW DO PEOPLE BEHAVE DIFFERENTLY?")
print("=" * 80)

print("\n1. POST LENGTH & DETAIL:")
print("-" * 60)
print(f"  Most detailed:  r/AskDocs (1,648 chars) - need to describe symptoms")
print(f"  Professional:   r/medicine (1,231 chars) - detailed case discussions")
print(f"  Emotional:      r/HealthAnxiety (972 chars) - anxiety narratives")
print(f"  Brief:          r/Health (89 chars) - just news headlines")
print(f"  → People write MORE when seeking personal medical help")

print("\n2. ENGAGEMENT PATTERNS:")
print("-" * 60)
print(f"  Highest engagement:  r/medicine (1,359 score, 192 comments)")
print(f"  Second:              r/Health (1,085 score, 117 comments)")
print(f"  Personal health:     r/AskDocs (514 score, 56 comments)")
print(f"  LOWEST:              r/medicaladvice (2 score, 1 comment)")
print(f"  → Professional/news content gets MORE engagement than personal health")

print("\n3. QUESTION VS STATEMENT BEHAVIOR:")
print("-" * 60)
print(f"  Most questions:  r/medicaladvice (62.8%) - desperate for answers")
print(f"  Questions:       r/AskDocs (41.7%) - seeking diagnosis")
print(f"  Statements:      r/medicine (14.8% questions) - sharing knowledge")
print(f"  → Advice-seeking communities ask more, professional communities share more")

print("\n4. EMOTIONAL LANGUAGE:")
print("-" * 60)
print(f"  Most anxious:    r/HealthAnxiety (0.86 anxiety words) - in the name!")
print(f"  Most urgent:     r/AskDocs (0.93 urgency words) - health concerns")
print(f"  Most positive:   r/biohackers (0.062 polarity) - optimizing health")
print(f"  Most neutral:    r/Health (0.01 anxiety) - just news")
print(f"  → Anxiety-focused communities use more emotional language")

print("\n5. MEDICAL TERMINOLOGY:")
print("-" * 60)
print(f"  Most medical:    r/medicine (1.72% density) - professionals")
print(f"  Patient trying:  r/AskDocs (1.05%) - using medical terms to sound credible")
print(f"  Casual:          r/biohackers (0.58%) - informal optimization talk")
print(f"  → Professionals use most medical jargon, patients try to mimic it")

print("\n" + "=" * 80)
print("INSIGHT 3: CAN WE PREDICT SUBREDDIT FROM CONTENT?")
print("=" * 80)

print("\nYES! Machine learning achieved 79.8% accuracy")
print("\nWhat this means:")
print("  • Each subreddit has DISTINCT language patterns")
print("  • People write differently based on their goal")
print("  • AI can identify community purpose from text alone")

print("\nMost distinctive words per subreddit:")
print("-" * 60)
print("  r/AskDocs:        'thank', 'doctor', 'male', 'female' (patient demographics)")
print("  r/medicine:       'patient', 'hospital', 'healthcare' (professional terms)")
print("  r/HealthAnxiety:  'anxiety', 'google', 'fear' (emotional terms)")
print("  r/Health:         'trump', 'fda', 'cdc' (political/news terms)")
print("  r/biohackers:     'supplements', 'creatine', 'aging' (optimization terms)")
print("  r/medicaladvice:  'advice', 'worried', 'help' (urgent help-seeking)")

print("\n" + "=" * 80)
print("INSIGHT 4: CLUSTERING REVEALS NATURAL GROUPINGS")
print("=" * 80)

print("\nSubreddits naturally cluster into 3 FUNCTIONAL GROUPS:")
print("\nGROUP 1: PERSONAL HEALTH SEEKING")
print("  • r/AskDocs (69% in cluster 4)")
print("  • r/medicaladvice")
print("  • r/HealthAnxiety (89% in cluster 0)")
print("  → People seeking help for personal symptoms")

print("\nGROUP 2: PROFESSIONAL/NEWS")
print("  • r/medicine (78% in cluster 1)")
print("  • r/Health (69% in cluster 3)")
print("  → Professionals and news consumers")

print("\nGROUP 3: OPTIMIZATION/EXPERIMENTATION")
print("  • r/biohackers (spread across clusters)")
print("  → Unique community focused on enhancement, not problems")

print("\n" + "=" * 80)
print("INSIGHT 5: SENTIMENT REVEALS COMMUNITY TONE")
print("=" * 80)

print("\nEmotional tone differs significantly:")
print("-" * 60)
print("  POSITIVE communities:")
print("    • r/biohackers (0.062) - excited about optimization")
print("    • r/medicine (0.060) - professional, solution-focused")
print("    • r/HealthAnxiety (0.048) - supportive community")

print("\n  NEUTRAL/NEGATIVE communities:")
print("    • r/AskDocs (0.026) - concerned about symptoms")
print("    • r/Health (0.012) - neutral news reporting")
print("    • r/medicaladvice (-0.000) - worried, seeking help")

print("\n" + "=" * 80)
print("FINAL ANSWER TO QUESTION 8")
print("=" * 80)

answer = """
Q: How do user behaviors differ across subreddits?

ANSWER: User behaviors differ DRAMATICALLY across medical subreddits based on 
their primary goal:

1. DIAGNOSIS-SEEKING (r/AskDocs, r/medicaladvice)
   - Write long, detailed symptom descriptions
   - Use urgent language and many questions
   - Attempt self-diagnosis (63.3% in AskDocs)
   - Try to use medical terminology to sound credible
   - Anxious and concerned tone

2. PROFESSIONAL DISCUSSION (r/medicine)
   - Share knowledge and discuss cases
   - Use most medical terminology (1.72% density)
   - Highest engagement (1,359 avg score)
   - Mostly statements, not questions
   - Professional, solution-focused tone

3. EMOTIONAL SUPPORT (r/HealthAnxiety)
   - Seek reassurance and validation
   - Highest anxiety language (0.86 words/post)
   - Most subjective content (0.503 score)
   - Google symptoms and fear worst outcomes
   - Supportive community tone

4. NEWS/INFORMATION (r/Health)
   - Share health news and articles
   - Shortest posts (just headlines)
   - No personal health issues (1.9% self-diagnosis)
   - Focus on politics, outbreaks, public health
   - Neutral, informational tone

5. OPTIMIZATION (r/biohackers)
   - Experiment with supplements and lifestyle
   - Most positive sentiment (0.062)
   - Data-driven discussions
   - Focus on enhancement, not problems
   - Curious, experimental tone

6. QUICK ADVICE (r/medicaladvice)
   - Brief questions seeking immediate help
   - Highest question rate (62.8%)
   - LOWEST engagement (mostly ignored)
   - Desperate, worried tone
   - Small, inactive community

KEY INSIGHT: The same health topic generates completely different behaviors
depending on whether users seek diagnosis, support, knowledge, or optimization.
Machine learning can distinguish these communities with 79.8% accuracy based
purely on how people write.
"""

print(answer)

# Save comprehensive report
with open('q8_final_answer.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("RESEARCH QUESTION 8: FINAL ANSWER\n")
    f.write("=" * 80 + "\n\n")
    f.write(answer)
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("DETAILED EVIDENCE\n")
    f.write("=" * 80 + "\n\n")
    
    for subreddit, data in purposes.items():
        f.write(f"\nr/{subreddit}: {data['purpose']}\n")
        f.write("-" * 60 + "\n")
        for evidence in data['evidence']:
            f.write(f"  • {evidence}\n")

print("\nSaved: q8_final_answer.txt")

print("\n" + "=" * 80)
print("QUESTION 8 COMPLETE WITH INSIGHTS!")
print("=" * 80)

