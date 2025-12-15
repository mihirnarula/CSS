# The Internet as a Doctor 🩺

**A Data-Driven Analysis of Medical Information-Seeking Behavior on Reddit**

This repository contains the complete data analysis and machine learning pipeline for our project studying **online self-diagnosis behavior on Reddit medical communities**. Using large-scale Reddit data (2011–2025), we analyze symptom descriptions, emotional states, self-diagnosis patterns, advice contradictions, and subreddit-level behavioral differences.

> 📊 **Dataset size:** 312,342 comments from 3,573 posts across 7 medical subreddits

---

## 📌 Project Overview

Online health communities have become a primary source of medical information for many users. This project investigates:

* How people describe symptoms online
* How often users attempt self-diagnosis
* Emotional patterns in health-seeking behavior
* Reliability and contradiction in peer-provided medical advice
* Differences in behavior across medical subreddits
* Whether posts discuss **acute** or **chronic** conditions

The project combines **NLP**, **machine learning**, and **network analysis** to study these questions at scale.

---

## 🧠 Research Questions

1. What symptoms are most commonly described online?
2. How prevalent are explicit self-diagnosis attempts?
3. What emotions dominate health-related posts?
4. Which conditions are most frequently self-diagnosed?
5. Do highly upvoted responses sound more medically authoritative?
6. Does urgency language affect response time?
7. Are users seeking new information or validation?
8. How do behaviors differ across medical subreddits?
9. Can we distinguish acute vs. chronic condition discussions?

---

## 🗂 Subreddits Analyzed

* `r/AskDocs`
* `r/medicine`
* `r/Health`
* `r/biohackers`
* `r/medical`
* `r/medicaladvice`
* `r/DiagnoseMe`

---

## ⚙️ Methodology

### 1. Data Collection

* Data collected using the **Reddit API**
* Time span: **2011–2025**
* Final dataset: **312,342 comments**

### 2. Data Preprocessing

Implemented in `clean_data.py`:

* Removed deleted / short / null entries
* Text normalization (lowercasing, URL removal, punctuation cleanup)
* Tokenization using **NLTK**
* Stopword removal (medical terms preserved)
* Lemmatization using **WordNet**
* Metadata extraction (timestamps, subreddit labels, engagement metrics)

Validation scripts:

* `explore_csv.py`
* `show_cleaned_summary.py`

### 3. Symptom & Condition Extraction

Implemented in `q1_symptom_analysis.py`:

* Hybrid rule-based + NER approach
* Symptom dictionary (500+ symptoms from SNOMED CT & MedDRA)
* Condition lexicon (200+ diseases)
* Co-occurrence network analysis
* Body-system categorization

### 4. Self-Diagnosis Detection

Implemented in `q2_self_diagnosis_detection.py`:

* Regex-based phrase detection (e.g., "I think I have…")
* Post-level categorization of diagnosis-seeking behavior

### 5. Emotional State Analysis

Implemented in `q3_emotion_analysis.py`:

* Sentiment analysis using **VADER**
* Emotion classification using **NRC Emotion Lexicon**
* Correlation of emotions with engagement metrics

### 6. Contradiction Detection

Implemented in `advanced_contradiction_analysis.py`:

* Advice extraction ("you should", "go to ER", etc.)
* Semantic similarity using sentence embeddings
* Negation & antonym detection
* Severity conflict detection (urgent vs non-urgent)

### 7. Machine Learning Models

#### Acute vs Chronic Classifier (`q9_chronic_vs_acute.py`)

* Features:

  * TF-IDF (5,000 features)
  * Temporal indicators
  * Readability & post length
  * Engagement metrics
* Models tested:

  * Logistic Regression
  * SVM (RBF)
  * Random Forest
  * Gradient Boosting

**Best accuracy:** 83.1% (Random Forest)

#### Post Intent Classifier (`q8_ai_models.py`)

* Classes:

  * Information-seeking
  * Validation-seeking
  * Support-seeking
  * Knowledge-sharing

### 8. Clustering & Visualization

* K-means clustering (k=5)
* Elbow method for cluster selection
* t-SNE for 2D visualization
* Cluster interpretation via top terms & engagement

### 9. Subreddit Comparison

Implemented in `q8_subreddit_comparison.py`:

* Terminology density
* Emotional tone
* Engagement patterns
* Self-diagnosis rates

---

## 📈 Key Findings

* **40.5%** of posts mention explicit symptoms
* **Mental health** symptoms dominate discussions
* **89%** of advice-seeking posts receive contradictory recommendations
* Only **2%** of users seek validation — most want new information
* Acute condition posts receive **31% more engagement** than chronic ones
* Medical jargon and citations do **not** correlate with higher-quality responses

---

## 📁 Repository Structure (Example)

```
.
├── data/                               
│   ├── all_subreddits.csv
│   ├── all_subreddits_cleaned.csv
│   ├── advanced_contradiction_results.csv
│   ├── case_study_top_10_posts.csv
│   ├── q1_symptom_features.csv
│   ├── q2_posts_analyzed.csv
│   ├── q2_post_comment_patterns.csv
│   ├── q2_self_diagnosis_features.csv
│   ├── q8_posts_with_features.csv
│   ├── q8_engagement_stats.csv
│   ├── q8_ai_sentiment_analysis.csv
│   ├── q8_ai_model_comparison.csv
│   ├── q8_subreddit_profiles.csv
│   ├── q9_chronic_acute_predictions.csv
│   └── q9_model_comparison.csv
│
├── clean_data.py                       # Data cleaning & preprocessing pipeline
├── explore_csv.py                     # Dataset exploration & sanity checks
├── show_cleaned_summary.py            # Summary statistics of cleaned data
├── full_dataset_analysis.py           # End-to-end dataset analysis
│
├── q1_symptom_analysis.py             # Symptom & condition extraction
├── q1_visualizations.py               # Symptom-related visualizations
├── q1_final_report.py                 # Q1 final analysis script
│
├── q2_self_diagnosis_detection.py     # Self-diagnosis detection logic
├── q2_proper_analysis.py              # Detailed Q2 analysis
│
├── q8_ai_models.py                    # Post intent classification & clustering
├── q8_subreddit_comparison.py         # Cross-subreddit behavioral analysis
├── q8_visualizations.py               # Subreddit comparison visualizations
│
├── q9_chronic_vs_acute.py             # Acute vs chronic condition classifier
├── q9_final_insights.py               # Model interpretation & insights
│
├── advanced_contradiction_analysis.py # Medical advice contradiction detection
├── case_study_top_posts.py            # Qualitative case-study analysis
│
├── create_methodology_visual.py       # Methodology diagram generation
│
├── METHODOLOGY.txt                    # High-level methodology explanation
├── project.txt                        # Project overview
├── research_ques.txt                  # Research questions
│
├── README.md                          # Project documentation
├── .gitignore
└── .gitattributes                     # Git LFS configuration

```

---

## 🛠️ Tech Stack

* **Python**
* **NLTK**, **spaCy**
* **scikit-learn**
* **Pandas**, **NumPy**
* **Matplotlib / Seaborn**
* **Reddit API**

---

## 👥 Team

* Aman Jaiswal
* Mihir Narula
* Yuven Blowria

Plaksha University

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. The analysis does **not** validate medical advice or diagnoses. Online health forums should **not** be used as a substitute for professional medical consultation.

---

## ⭐ Citation

If you use this work, please cite our project report:

> *The Internet as a Doctor: How People Self-diagnose Online – A Data-Driven Analysis of Medical Information-Seeking Behavior on Reddit*
