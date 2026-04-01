#  Employee Analytics — 100 Pandas & NumPy Exercises
### From Data Cleaning to Machine Learning Preparation

> Inspired by the legendary [numpy-100](https://github.com/rougier/numpy-100) and [pandas-exercises](https://github.com/guipsamora/pandas_exercises) repositories.
> This project takes a different approach — instead of isolated snippets, every exercise builds on the previous one, simulating a **real data science workflow** from raw data to ML-ready dataset.

---

##  Dataset

**1,010 employees** across **8 countries**, **7 departments**, and **25 columns** covering demographics, compensation, performance, and work style.

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/employees_practice.csv', parse_dates=['hire_date', 'last_promotion_date'])
df.head()
```

| Column | Description |
|---|---|
| `employee_id` | Unique identifier (e.g. EMP1496) |
| `first_name`, `last_name` | Employee name |
| `email` | Work email |
| `gender` | Female / Male / Non-binary |
| `age` | Age in years |
| `city`, `country` | Location |
| `department` | HR / Sales / Engineering / Finance / Marketing / Operations / Legal |
| `job_education` | High School / Bachelor / Master / PhD |
| `contract_type` | Full-time / Part-time / Contract |
| `status` | Active / Resigned / Terminated / On Leave |
| `hire_date` | Date of joining |
| `years_experience` | Total years of experience |
| `salary`, `bonus` | Annual compensation |
| `performance_rating` | 1 to 5 (5 = best) |
| `projects_completed` | Number of completed projects |
| `remote_days_per_week` | 0 to 5 |
| `satisfaction_score` | 1 to 10 (10 = happiest) |
| `training_hours` | Hours in training |
| `certifications` | Number of certifications |
| `is_manager`, `is_dept_head` | Boolean leadership flags |
| `last_promotion_date` | Date of last promotion |

---

## Part 1 — NumPy Fundamentals

#### 1.1
Extract the `salary` column as a NumPy array (drop NaNs first). Compute the mean, median, standard deviation, min, and max using **only NumPy functions** — no pandas allowed.

---

#### 1.2
Normalize the salary array to a 0–1 scale using the min-max formula:
```
normalized = (x - min) / (max - min)
```
Use pure NumPy with no loops. Verify the result has min=0.0 and max=1.0.

---

#### 1.3
Use `np.where()` to classify each salary into one of three categories:
- Below 80,000 → `"Low"`
- Between 80,000 and 110,000 → `"Mid"`
- Above 110,000 → `"High"`

Store the result in `salary_level`. Print the first 10 values and the count of each category.

---

#### 1.4
Use `np.percentile()` to compute the 25th, 50th, 75th, and 90th percentile of salaries. Format the output with dollar signs and comma separators.

---

#### 1.5
Create a 2D NumPy array of shape `(5, 3)` with random integers between 50,000 and 150,000 — simulating 5 employees × 3 months of salary. Compute the mean salary per employee and per month.

---

## Part 2 — Loading & Inspection

#### 2.1
Display the shape, column names as a list, dtypes, and first 5 rows.

---

#### 2.2
Show the count and percentage of missing values per column. Show only columns that have missing values, sorted by most missing first.

---

#### 2.3
For these columns: `gender`, `department`, `contract_type`, `status`, `job_education`, `city`, `country` — show how many unique values each has and list the actual values.

---

#### 2.4
Check for fully duplicate rows. Report how many exist and display them.

---

#### 2.5
What is the earliest and latest `hire_date` in the dataset? How many years does the hiring history span?

---

## Part 3 — Data Cleaning

#### 3.1
Standardize all column names: lowercase, strip whitespace, replace spaces with underscores.

---

#### 3.2
Fill missing `salary` values with the median salary of that employee's **department** — not the global median.

---

#### 3.3
Fill missing `satisfaction_score` with the column mean, rounded to 1 decimal place.

---

#### 3.4
Fill missing `city` and `country` with `"Unknown"` in a single operation.

---

#### 3.5
Fill missing `last_promotion_date` with that employee's own `hire_date`.

---

#### 3.6
Verify that `is_manager` and `is_dept_head` are proper boolean dtype. Convert them if they aren't.

---

#### 3.7
Remove rows where **both** `salary` AND `bonus` are missing simultaneously.

---

#### 3.8 ✅ Verification
Verify zero NaNs remain and print the new shape of the DataFrame.

---

## Part 4 — Feature Engineering

#### 4.1
Create a `total_compensation` column = `salary` + `bonus`.

---

#### 4.2
Create a `bonus_pct` column = bonus as a percentage of salary, rounded to 2 decimal places.

---

#### 4.3
Create a `years_since_promotion` column = years between `last_promotion_date` and today.

---

#### 4.4
Create a `tenure_years` column = years the employee has been with the company.

---

#### 4.5
Create a `seniority` column using `pd.cut()` on `tenure_years`:
- 0–2 → `"Junior"`, 2–5 → `"Mid"`, 5–10 → `"Senior"`, 10+ → `"Lead"`

---

#### 4.6
Create a `salary_band` column using `pd.qcut()` on `salary` into 4 equal quantile groups: `Q1`, `Q2`, `Q3`, `Q4`.

---

#### 4.7
Create a boolean column `is_remote` = `True` if `remote_days_per_week` >= 3.

---

#### 4.8
From `hire_date`, extract: `hire_year`, `hire_month`, and `hire_day_of_week` (as day name).

---

## Part 5 — Filtering & Selection

#### 5.1
Select all Active, Full-time employees in the Engineering department.

---

#### 5.2
Find employees who are managers but NOT department heads, earning more than 100,000.

---

#### 5.3
Find employees hired before January 1st, 2010 who are still Active.

---

#### 5.4
Find the top 10 highest paid employees. Show only: `first_name`, `last_name`, `department`, `salary`.

---

#### 5.5
Find employees with `performance_rating` of 5 AND `satisfaction_score` below 5. How many are there? Write a markdown cell with your interpretation of this pattern.

---

#### 5.6
Use `.query()` to find all Non-binary employees in Sales or Marketing with salary above 85,000.

---

## Part 6 — GroupBy & Aggregation

#### 6.1
For each department compute: headcount, mean salary, max salary, min salary, median bonus. Sort by mean salary descending.

---

#### 6.2
For each country, compute the percentage breakdown of employee status. Each country's percentages should sum to 100%.

---

#### 6.3
For each `contract_type` + `seniority` combination, compute average `performance_rating`. Unstack the result so seniority levels become columns.

---

#### 6.4
Use `.transform()` to add:
- `dept_avg_salary` = mean salary of the employee's department
- `salary_vs_dept_avg` = difference between employee salary and their dept average

---

#### 6.5
Keep only employees from departments where the average satisfaction score is above 5. How many employees are removed?

---

#### 6.6
Using named aggregations, create a clean department summary with: `headcount`, `avg_salary`, `avg_bonus`, `avg_performance`, `avg_satisfaction`.

---

## Part 7 — Sorting, Ranking & Window Functions

#### 7.1
Rank employees within each department by salary (highest = rank 1). Add as `salary_rank_in_dept`.

---

#### 7.2
Sort by `department` A→Z, then by `salary` highest first within each department.

---

#### 7.3
Sort by `hire_date` and compute the cumulative total salary spend over time.

---

#### 7.4
Compute a 3-employee rolling average of salary (sorted by hire_date). What trend do you notice? Write a markdown cell with your observation.

---

## Part 8 — String Operations

#### 8.1
Extract the domain from the `email` column. Verify all emails use `company.com`.

---

#### 8.2
Create a `full_name` column combining `first_name` and `last_name`.

---

#### 8.3
Ensure all values in `gender` are in Title Case.

---

#### 8.4
Find all employees whose `last_name` starts with `"M"`. How many are there?

---

#### 8.5
Count how many employees have a `first_name` longer than 5 characters.

---

## Part 9 — MultiIndex & Pivot Tables

#### 9.1
Create a pivot table showing average salary by `department` (rows) and `contract_type` (columns). Fill missing with 0.

---

#### 9.2
Create a pivot table showing headcount by `country` and `gender`. Include row and column totals.

---

#### 9.3
Group by `country` + `department` and compute average salary. Use `.xs()` to extract one country's data.

---

#### 9.4
Use `pd.crosstab()` to show `performance_rating` distribution across `seniority` levels. Normalize by row to show percentages.

---

## Part 10 — Combining DataFrames

#### 10.1
Split into `df_active` and `df_inactive`, concatenate back, and verify the shape matches the original.

---

#### 10.2
Merge this budget table with the main DataFrame and compute what percentage of each department's budget goes to salaries:

```python
dept_budget = pd.DataFrame({
    'department': ['Engineering', 'Sales', 'HR', 'Finance',
                   'Marketing', 'Operations', 'Legal'],
    'annual_budget': [5_000_000, 3_000_000, 1_500_000,
                      2_000_000, 1_800_000, 2_500_000, 1_200_000]
})
```

---

## Part 11 — Final Business Analysis

#### 11.1 — Attrition by Department
Which department has the highest attrition rate? (Resigned + Terminated as % of total headcount)

---

#### 11.2 — Experience vs Salary
Is there a correlation between `years_experience` and `salary`? Compute the Pearson correlation and interpret it.

---

#### 11.3 — Managers vs Non-Managers
Do managers earn significantly more? Compute the salary gap in absolute and percentage terms.

---

#### 11.4 — Education vs Performance
Which `job_education` level has the highest average `performance_rating`?

---

#### 11.5 — Promotion Speed
What is the average time to first promotion (in years)? Which department promotes fastest?

---

#### 11.6 — CEO Dashboard ⭐
Create a department summary with: `headcount`, `avg_salary`, `attrition_rate`, `avg_performance`, `avg_satisfaction`, `avg_tenure`. Export to `outputs/dept_summary.csv`.

---

## Part 12 — ML Preparation 🤖

#### 12.1
Create a binary target column `attrition`: `1` if Resigned or Terminated, `0` otherwise.

---

#### 12.2
Drop columns that leak the target or are irrelevant: `employee_id`, `first_name`, `last_name`, `email`, `status`, `last_promotion_date`, `hire_date`.

---

#### 12.3
One-hot encode all remaining categorical columns using `pd.get_dummies()` with `drop_first=True`.

---

#### 12.4
Verify zero NaNs remain. Handle any that exist.

---

#### 12.5
Split into `X` (features) and `y` (target). Print both shapes.

---

#### 12.6
Apply min-max normalization to all numeric columns in `X` simultaneously.

---

#### 12.7 ✅ Final Check
Print `X.shape`, `y.value_counts()`, and confirm zero NaNs. Your data is ready for scikit-learn.

```python
# What comes next:
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

**You just built an end-to-end data science pipeline. Welcome to ML.** 🎉

---

## 📊 Skills Covered

| Skill | Parts |
|---|---|
| NumPy arrays & vectorization | 1 |
| Data inspection & profiling | 2 |
| Missing value strategies | 3 |
| Feature creation & date parsing | 4 |
| Boolean indexing & query | 5 |
| GroupBy, transform, filter | 6 |
| Ranking & window functions | 7 |
| String operations | 8 |
| Pivot tables & MultiIndex | 9 |
| Concat & merge | 10 |
| Business analysis & storytelling | 11 |
| ML preprocessing pipeline | 12 |
---

*If this helped you, give it a ⭐ — it means a lot.*
