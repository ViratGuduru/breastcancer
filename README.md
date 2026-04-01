# 🎗️ Breast Cancer Diagnosis Analysis
### Healthcare Data Analytics Portfolio Project

**Author:** Virat  
**Tools:** Python · Matplotlib · Seaborn · Scipy · Scikit-learn · AI-assisted analysis  
**Dataset:** Breast Cancer Research — 1,200 patient records, 21 features  
**Live Dashboard:** [https://viratguduru.github.io/breastcancer/](https://viratguduru.github.io/breastcancer/)

---

## 📁 Files in This Repository

| File | Description | How to Open |
|------|-------------|-------------|
| `index.html` | Interactive web dashboard | Any browser — no software needed |
| `breast_cancer_analysis.py` | Full Python analysis script | Run with Python 3.8+ |
| `README.md` | Project documentation | GitHub renders automatically |

---

## 🌐 Live Dashboard

👉 **https://viratguduru.github.io/breastcancer/**

Opens in any browser — no software required. Built with Chart.js.

---

## 🐍 Python Analysis

The entire analysis was performed in **Python** using the following stack:

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, aggregation |
| `numpy` | Numerical computations, effect sizes |
| `matplotlib` | Dashboard layout and all chart rendering |
| `seaborn` | Correlation heatmap |
| `scipy.stats` | T-tests and statistical significance |
| `scikit-learn` | StandardScaler for feature normalisation |

### How to Run
```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# 2. Place the CSV in the same folder as the script
# 3. Run
python breast_cancer_analysis.py
```

**Output:** `breast_cancer_dashboard.png` — a 7-panel analytical dashboard saved at 180 DPI.

---

## 🤖 AI-Assisted Analysis

This project was developed with the assistance of **Claude (Anthropic)** as a senior health data analyst AI collaborator. The AI contributed to:

- Defining the research questions and analytical framework
- Identifying the most clinically relevant features to analyse
- Engineering the composite **Malignancy Risk Score** formula
- Interpreting counter-intuitive findings (e.g. benign tumors averaging larger area)
- Writing clean, well-documented Python code
- Generating the interactive HTML dashboard
- Structuring findings into clinically meaningful narratives

---

## ❓ Research Questions Answered

| # | Question | Key Finding |
|---|----------|-------------|
| 1 | What is the diagnosis distribution? | 49% Malignant / 51% Benign — nearly balanced |
| 2 | Which features best separate M vs B? | `area_mean` leads with effect size d = 0.155 |
| 3 | Does tumor size predict malignancy? | No — benign tumors average *larger* area (652 mm² vs 629 mm²) |
| 4 | Do worst-case features outperform means? | Yes — `_worst` features consistently more discriminating |
| 5 | Can a composite score improve separation? | Yes — weighted Risk Score outperforms any single feature |

---

## 📊 Dashboard Panels

| Panel | Chart Type | Insight |
|-------|-----------|---------|
| Diagnosis Distribution | Horizontal bar | 612 Benign / 588 Malignant |
| Feature Discriminative Power | Effect size bars | Top 10 features ranked by Cohen's d |
| Avg Feature Values | Grouped bar (normalised) | M vs B comparison across 6 key features |
| Radius vs Concavity | Scatter plot | Significant class overlap confirmed |
| Radius Worst Distribution | Histogram | Frequency distribution by diagnosis |
| Malignancy Risk Score | Bar + error bars + significance | Composite score with p-value bracket |
| Correlation Matrix | Heatmap | Multicollinearity among worst-case features |

---

## 🔑 Top 3 Clinical Findings

1. **Size ≠ Malignancy** — Benign tumors average a *larger* area than malignant ones. Morphological shape metrics matter more than size alone.
2. **Worst-case beats average** — The `_worst` suffix features consistently outperform `_mean` features. Extreme cellular measurements capture the irregular geometry of cancerous tissue.
3. **No single threshold is enough** — The scatter plot confirms significant class overlap. Multi-feature models and composite scoring are clinically essential.

---

## 📚 Data Source

[Kaggle — guriya79/breast-cancer](https://www.kaggle.com/datasets/guriya79/breast-cancer/data)  
Original: Breast Cancer Wisconsin (Diagnostic) Dataset, UCI Machine Learning Repository
---