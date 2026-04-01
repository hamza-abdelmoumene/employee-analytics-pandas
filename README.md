# Employee Analytics — Pandas & NumPy Capstone

> A real-world data science project simulating a complete analytics workflow — from raw messy data to a machine learning-ready dataset.

---

## About

Most pandas exercises online are isolated snippets that don't connect to each other.

This project is different. Every exercise builds on the previous one, working on the **same dataset** throughout — exactly like a real data science job.

You'll go from a raw CSV with missing values and messy columns, all the way to a clean, encoded, normalized dataset ready to feed into a machine learning model.

---

##  Dataset

**1,010 employees** | **8 countries** | **7 departments** | **25 columns**

The dataset covers employee demographics, compensation, performance metrics, work style, and employment history — with intentional missing values, duplicates, and mixed types to simulate real-world data.

---

##Structure

```
employee-analytics-pandas/
├── data/
│   └── employees_practice.csv
├── notebooks/
│   └── employee_analytics.ipynb
├── outputs/
│   └── dept_summary.csv
├── EXERCISES.md              ← exercises without hints
├── EXERCISES_WITH_HINTS.md   ← exercises with hints
├── requirements.txt
└── README.md
```

---

##  Skills Covered

| Part | Topic |
|---|---|
| 1 | NumPy arrays, vectorization, percentiles |
| 2 | Data inspection, missing values, duplicates |
| 3 | Data cleaning, smart fills, boolean conversion |
| 4 | Feature engineering, date parsing, binning |
| 5 | Boolean indexing, query method |
| 6 | GroupBy, transform, filter, named aggregations |
| 7 | Sorting, ranking, cumulative & rolling functions |
| 8 | String operations, regex extraction |
| 9 | Pivot tables, MultiIndex, cross-tabulation |
| 10 | Concat, merge, SQL-style joins |
| 11 | Business analysis, storytelling with data |
| 12 | ML preprocessing, encoding, normalization |

---

##  Key Findings

- **Legal** has the highest average salary ($118K), while **HR** has the lowest ($89K)
- **~18%** of top performers (rating=5) are unsatisfied — a significant flight risk
- Strong positive correlation between `years_experience` and `salary` (r ≈ 0.71)
- Managers earn on average **23% more** than non-managers
- **Sales** has the highest attrition rate across all departments

---

##  How to Run

```bash
# Clone the repo
git clone https://github.com/hamza-abdelmoumene/employee-analytics-pandas.git
cd employee-analytics-pandas

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/employee_analytics.ipynb
```

---

##  Requirements

```
pandas
numpy
jupyter
scikit-learn
```

---

##  What's Next

After completing all 12 parts, the data is ML-ready:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

The next step is a dedicated ML project predicting employee attrition using this cleaned dataset.

---

## ✍️ Author

**Hamza Abdel**
- GitHub: [@hamza-abdel](https://github.com/hamza-abdel)

---

*If this helped you, give it a ⭐ — it means a lot and helps others find it.*