# Employee Analytics — 100 Pandas & NumPy Exercises
### From Data Cleaning to Machine Learning Preparation

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

> 💡 pandas Series has a `.to_numpy()` method. Then look for `np.mean`, `np.median`, `np.std`, `np.min`, `np.max`.

---

#### 1.2
Normalize the salary array to a 0–1 scale using the min-max formula:
```
normalized = (x - min) / (max - min)
```
Use **pure NumPy with no loops**. Verify the result has min=0.0 and max=1.0.

> 💡 NumPy operations are vectorized — you can apply math directly to the whole array just like a single number. Print the first 5 values and the min/max to verify.

---

#### 1.3
Use `np.where()` to classify each salary into one of three categories:
- Below 80,000 → `"Low"`
- Between 80,000 and 110,000 → `"Mid"`
- Above 110,000 → `"High"`

Store the result in an array called `salary_level`. Print the first 10 values and the count of each category.

> 💡 `np.where()` handles one condition at a time. **Nest** two `np.where()` calls to handle three categories. Use `np.unique(arr, return_counts=True)` to count each category.

---

#### 1.4
Use `np.percentile()` to compute the 25th, 50th, 75th, and 90th percentile of salaries. Format the output with dollar signs and comma separators.

> 💡 `np.percentile()` accepts a **list** of percentile values in a single call. Use f-strings with `{value:,.0f}` for formatting.

---

#### 1.5
Create a 2D NumPy array of shape `(5, 3)` with random integers between 50,000 and 150,000 — simulating 5 employees × 3 months of salary. Compute:
- Mean salary **per employee** (across months)
- Mean salary **per month** (across employees)

> 💡 `np.random.randint()` accepts a `size` parameter for 2D arrays. The `axis` parameter in `.mean()` controls the direction: `axis=1` = row-wise, `axis=0` = column-wise.

---

## Part 2 — Loading & Inspection

#### 2.1
Display the shape, column names as a list, dtypes, and first 5 rows. Keep each on a separate cell for clean notebook output.

> 💡 Use `.shape`, `.columns.tolist()`, `.dtypes`, and `.head()`. In Jupyter, the last line of a cell renders as a table automatically — no need for `print()`.

---

#### 2.2
Show the count **and** percentage of missing values per column. Show only columns that actually have missing values, sorted by most missing first.

> 💡 Build a DataFrame with two columns: one from `.isnull().sum()` and one from dividing by `len(df)` × 100. Filter with `> 0` to hide clean columns.

---

#### 2.3
For these columns: `gender`, `department`, `contract_type`, `status`, `job_education`, `city`, `country` — show how many unique values each has and list the actual values.

> 💡 Build a summary DataFrame with `.nunique()` for counts and `.unique()` for values. Use a list comprehension to apply `.unique()` across all columns at once.

---

#### 2.4
Check if there are any fully duplicate rows. Report how many exist and display them.

> 💡 `.duplicated()` marks the second occurrence as True by default. Use `keep=False` to mark **all** copies including the original. Count True values with `.sum()`.

---

#### 2.5
What is the earliest and latest `hire_date` in the dataset? How many years does the hiring history span?

> 💡 `.min()` and `.max()` work on datetime columns. Subtract them to get the span.

---

## Part 3 — Data Cleaning

#### 3.1
Standardize all column names: lowercase, strip whitespace, replace spaces with underscores.

> 💡 `df.columns` is a pandas Index — chain `.str.strip()`, `.str.lower()`, and `.str.replace(' ', '_')` on it directly.

---

#### 3.2
Fill missing `salary` values with the **median salary of that employee's department** — not the global median.

> 💡 `groupby('department')['salary'].transform('median')` returns a Series with the same index as `df`, where each value is the median of that row's department. Pass this into `.fillna()`.

---

#### 3.3
Fill missing `satisfaction_score` with the column mean, rounded to 1 decimal place.

> 💡 Chain `.mean()` and `.round(1)` together and pass the result into `.fillna()`.

---

#### 3.4
Fill missing `city` and `country` with the string `"Unknown"` in a single operation.

> 💡 `.fillna()` accepts a **dictionary** mapping column names to fill values. Assign the result back to `df[['city', 'country']]`.

---

#### 3.5
Fill missing `last_promotion_date` with that employee's own `hire_date`.

> 💡 `.fillna()` accepts another **Series** as the fill value — not just scalars. Pass `df['hire_date']` directly.

---

#### 3.6
Verify that `is_manager` and `is_dept_head` are proper boolean dtype. Convert them if they aren't.

> 💡 Check `.dtype` first. Use `.astype(bool)` to convert. Add a conditional print to report whether conversion was needed or not.

---

#### 3.7
Remove rows where **both** `salary` AND `bonus` are missing simultaneously.

> 💡 `dropna()` has a `subset` parameter for specific columns and a `how` parameter. `how='all'` drops only when **all** specified columns are NaN.

---

#### 3.8 ✅ Verification
After all cleaning steps, verify zero NaNs remain and print the new shape of the DataFrame.

> 💡 `df.isnull().sum().sum()` gives the total NaN count across all cells. An empty Series means you're clean.

---

## Part 4 — Feature Engineering

#### 4.1
Create a `total_compensation` column = `salary` + `bonus`.

> 💡 Direct column addition — assign the result back as a new column in `df`.

---

#### 4.2
Create a `bonus_pct` column = bonus as a percentage of salary, rounded to 2 decimal places.

> 💡 `(bonus / salary) * 100` — chain `.round(2)` at the end.

---

#### 4.3
Create a `years_since_promotion` column = number of years between `last_promotion_date` and today.

> 💡 `pd.Timestamp.now()` gives today. Subtracting two datetime columns gives a Timedelta. Use `.dt.days` to get days, then divide by 365.25 for years.

---

#### 4.4
Create a `tenure_years` column = years the employee has been with the company (from `hire_date` to today).

> 💡 Same approach as 4.3 but using `hire_date`. Store as a column in `df`.

---

#### 4.5
Create a `seniority` column using `pd.cut()` on `tenure_years`:
- 0–2 years → `"Junior"`
- 2–5 years → `"Mid"`
- 5–10 years → `"Senior"`
- 10+ years → `"Lead"`

> 💡 `pd.cut()` needs `bins` and `labels`. Use `np.inf` for the open upper bound. The number of labels must equal the number of intervals (bins - 1).

---

#### 4.6
Create a `salary_band` column using `pd.qcut()` on `salary` into 4 equal quantile groups labeled `Q1`, `Q2`, `Q3`, `Q4`.

> 💡 `pd.qcut()` splits by **equal-sized groups** automatically — unlike `pd.cut()` which splits by value ranges. Use `q=4` and pass a `labels` list.

---

#### 4.7
Create a boolean column `is_remote` = `True` if `remote_days_per_week` >= 3.

> 💡 A comparison on a column already returns a boolean Series — no `if/else` needed.

---

#### 4.8
From `hire_date`, extract three new columns: `hire_year`, `hire_month`, and `hire_day_of_week` (as day name, e.g. `"Monday"`).

> 💡 The `.dt` accessor exposes date components. Use `.dt.year`, `.dt.month`, and `.dt.day_name()`.

---

## Part 5 — Filtering & Selection

#### 5.1
Select all **Active, Full-time** employees in the **Engineering** department.

> 💡 Three conditions combined with `&`. Wrap each condition in parentheses. Match exact capitalization — pandas is case-sensitive.

---

#### 5.2
Find employees who are **managers but NOT department heads**, earning more than 100,000.

> 💡 Use `~` to negate a boolean column. Three conditions combined with `&`.

---

#### 5.3
Find employees hired **before January 1st, 2010** who are **still Active**.

> 💡 You can compare a datetime column directly to a date string like `'2010-01-01'` — pandas handles the conversion automatically.

---

#### 5.4
Find the top 10 highest paid employees. Show only: `first_name`, `last_name`, `department`, `salary`.

> 💡 `.sort_values()` descending + `.head(10)` + column selection with a list.

---

#### 5.5
Find employees with `performance_rating` of 5 AND `satisfaction_score` below 5. How many are there? Write a markdown cell interpreting what this pattern suggests about the company.

> 💡 Two conditions with `&`. Use `len()` to count rows — not `.size` which counts total cells. Think about what it means to be a top performer who is unhappy.

---

#### 5.6
Use `.query()` to find all Non-binary employees in Sales or Marketing with a salary above 85,000.

> 💡 Inside `.query()` use `and`/`or` instead of `&`/`|`. For list membership use `in [...]`. Use double quotes outside, single quotes inside for string values.

---

## Part 6 — GroupBy & Aggregation

#### 6.1
For each department compute: headcount, mean salary, max salary, min salary, and median bonus. Sort by mean salary descending. Round all numeric results to 2 decimal places.

> 💡 Use named aggregation syntax: `.agg(new_name=('column', 'function'))`. This lets you aggregate different columns with different functions in one clean call.

---

#### 6.2
For each country, compute the percentage breakdown of employee status (Active / Resigned / Terminated / On Leave). Each country's percentages should sum to 100%.

> 💡 `groupby('country')['status'].value_counts(normalize=True) * 100` — `normalize=True` converts counts to proportions automatically.

---

#### 6.3
For each `contract_type` + `seniority` combination, compute the average `performance_rating`. Unstack the result so seniority levels become columns — making it easy to compare across contract types.

> 💡 `groupby(['col1', 'col2'])` creates a MultiIndex result. `.unstack()` pivots the inner index level into columns. Chain `.round(2)` for clean output.

---

#### 6.4
Use `.transform()` to add two new columns:
- `dept_avg_salary` = mean salary of the employee's department
- `salary_vs_dept_avg` = how much above or below the employee is from their department average

> 💡 `groupby('department')['salary'].transform('mean')` returns a Series the same length as `df` — each value is the mean of that row's group. This is what makes `transform()` different from `agg()`.

---

#### 6.5
Keep only employees from departments where the **average satisfaction score is above 5**. How many employees are removed?

> 💡 `groupby().filter()` takes a lambda that receives each group as a DataFrame and returns True/False. The result keeps only rows from groups where the condition is True.

---

#### 6.6
Using named aggregations, create a clean department summary DataFrame with columns: `headcount`, `avg_salary`, `avg_bonus`, `avg_performance`, `avg_satisfaction`. This should be readable enough to hand to an HR director.

> 💡 Named agg syntax: `.agg(output_name=('source_column', 'aggregation_function'))`. Chain `.round(2)` at the end.

---

## Part 7 — Sorting, Ranking & Window Functions

#### 7.1
Rank employees **within each department** by salary, where the highest earner gets rank 1. Add as a new column `salary_rank_in_dept`.

> 💡 `groupby('department')['salary'].rank(ascending=False)` assigns ranks per group automatically. The result has the same index as `df`.

---

#### 7.2
Sort the DataFrame by `department` alphabetically (A→Z), then by `salary` highest first within each department.

> 💡 `sort_values()` accepts a **list** of column names and a corresponding list of `ascending` booleans.

---

#### 7.3
Sort employees by `hire_date` and compute the **cumulative total salary spend** as if you hired them one by one over time.

> 💡 Sort first, then `.cumsum()` on the salary column. The last value represents total salary spend across all employees.

---

#### 7.4
After sorting by `hire_date`, compute a **3-employee rolling average** of salary. Plot or print the first 20 values. What trend do you notice about how starting salaries changed over time?

> 💡 `.rolling(window=3).mean()` — the first two values will be NaN since there aren't enough preceding rows yet. Write a markdown cell with your observation.

---

## Part 8 — String Operations

#### 8.1
Extract the domain from the `email` column (everything after `@`). Verify that 100% of emails use `company.com` using `.value_counts()`.

> 💡 `.str.split('@')` returns a list per row. Use `.str[1]` to get the second element. Or use `.str.extract()` with a regex pattern like `r'@(.+)'`.

---

#### 8.2
Create a `full_name` column by combining `first_name` and `last_name` with a space in between.

> 💡 String columns support the `+` operator just like Python strings. Don't forget the space: `' '`.

---

#### 8.3
Ensure all values in the `gender` column are in Title Case (e.g. `"non-binary"` → `"Non-Binary"`).

> 💡 `.str.title()` — one line. Check `.value_counts()` before and after to confirm.

---

#### 8.4
Find all employees whose `last_name` starts with the letter `"M"`. How many are there?

> 💡 `.str.startswith('M')` returns a boolean Series — use it to filter the DataFrame.

---

#### 8.5
Count how many employees have a `first_name` longer than 5 characters.

> 💡 `.str.len()` returns a numeric Series of string lengths. Apply a comparison and count the True values.

---

## Part 9 — MultiIndex & Pivot Tables

#### 9.1
Create a pivot table showing the **average salary** by `department` (rows) and `contract_type` (columns). Fill missing combinations with 0. Round to 2 decimals.

> 💡 `pd.pivot_table()` with `values`, `index`, `columns`, `aggfunc='mean'`, and `fill_value=0`.

---

#### 9.2
Create a pivot table showing **headcount** by `country` and `gender`. Add row and column totals.

> 💡 Use `aggfunc='count'` and `margins=True` for totals. Pick any non-null column for `values`.

---

#### 9.3
Group by `country` + `department` and compute average salary. This creates a MultiIndex Series. Then use `.xs()` to extract just one country's data.

> 💡 `.xs('country_name', level='country')` slices a specific outer level. Pick any country that exists in your data.

---

#### 9.4
Use `pd.crosstab()` to show the distribution of `performance_rating` across `seniority` levels. Normalize **by row** so each seniority level sums to 100%. Round to 1 decimal.

> 💡 `normalize='index'` normalizes each row independently. Multiply by 100 for percentages.

---

## Part 10 — Combining DataFrames

#### 10.1
Split the DataFrame into `df_active` (status == Active) and `df_inactive` (all others). Concatenate them back together and verify the final shape matches the original.

> 💡 Use `pd.concat([df1, df2], ignore_index=True)` to reset the index after combining. Compare `.shape` before and after.

---

#### 10.2
Create this budget lookup table manually and merge it with the main DataFrame:

```python
dept_budget = pd.DataFrame({
    'department': ['Engineering', 'Sales', 'HR', 'Finance',
                   'Marketing', 'Operations', 'Legal'],
    'annual_budget': [5_000_000, 3_000_000, 1_500_000,
                      2_000_000, 1_800_000, 2_500_000, 1_200_000]
})
```

After merging, compute what **percentage of each department's budget** is consumed by total salaries.

> 💡 `pd.merge(df1, df2, on='department')` — then `groupby('department')['salary'].sum()` divided by `annual_budget` × 100.

---

## Part 11 — Final Business Analysis

#### 11.1 — Attrition by Department
Which department has the highest attrition rate? Define attrition as Resigned or Terminated employees. Express as a percentage of each department's total headcount.

> 💡 Filter for resigned/terminated, groupby department, count, then divide by total headcount per department × 100.

---

#### 11.2 — Experience vs Salary
Is there a correlation between `years_experience` and `salary`? Compute the Pearson correlation coefficient and interpret the result.

> 💡 `.corr()` between two columns. A value close to 1 = strong positive correlation. Write a markdown cell with your interpretation.

---

#### 11.3 — Managers vs Non-Managers
Do managers earn significantly more than non-managers on average? Compute the salary difference in absolute terms and as a percentage.

> 💡 `groupby('is_manager')['salary'].mean()` — then compute the difference and percentage gap between the two groups.

---

#### 11.4 — Education vs Performance
Which `job_education` level has the highest average `performance_rating`? Does more education mean better performance?

> 💡 `groupby('job_education')['performance_rating'].mean().sort_values(ascending=False)` — write a markdown cell with your take.

---

#### 11.5 — Promotion Speed
What is the average time between `hire_date` and `last_promotion_date` in years? Which department promotes its employees the fastest?

> 💡 Subtract the two date columns, convert to days with `.dt.days`, divide by 365.25. Then groupby department and take the mean.

---

#### 11.6 — CEO Dashboard ⭐
Create a single clean summary DataFrame — one row per department — containing: `headcount`, `avg_salary`, `attrition_rate`, `avg_performance`, `avg_satisfaction`, `avg_tenure`. Export it to `outputs/dept_summary.csv`.

> 💡 Compute each metric separately as a Series, then combine them with `pd.concat(axis=1)` or use one big `.agg()` call. This is the hardest exercise — take your time.

---

## Part 12 — ML Preparation 🤖

#### 12.1 — Target Column
Create a binary column `attrition`:
- `1` if status is `"Resigned"` or `"Terminated"`
- `0` otherwise

> 💡 `.isin(['Resigned', 'Terminated'])` returns a boolean Series. Cast to integer with `.astype(int)`.

---

#### 12.2 — Drop Leaky Columns
Drop columns that would leak the target or are irrelevant for prediction: `employee_id`, `first_name`, `last_name`, `email`, `status`, `last_promotion_date`, `hire_date`.

> 💡 `df.drop(columns=[...])` — assign back to `df`.

---

#### 12.3 — One-Hot Encoding
Encode all remaining categorical columns using `pd.get_dummies()`. Use `drop_first=True` to avoid the dummy variable trap.

> 💡 Pass a list of categorical column names to `pd.get_dummies(df, columns=[...], drop_first=True)`. Check `.dtypes` first to identify which columns are still `object` type.

---

#### 12.4 — Final NaN Check
Verify there are zero NaNs remaining in the dataset. If any exist, handle them before proceeding.

> 💡 `df.isnull().sum().sum()` should return exactly 0.

---

#### 12.5 — X and y Split
Split into features `X` (all columns except `attrition`) and target `y` (`attrition` column only). Print the shape of both.

> 💡 `X = df.drop(columns=['attrition'])` and `y = df['attrition']`.

---

#### 12.6 — Normalize Features
Apply min-max normalization to all numeric columns in `X` simultaneously.

> 💡 `df.select_dtypes(include=np.number)` selects numeric columns only. Apply the formula `(X - X.min()) / (X.max() - X.min())` — it broadcasts across all columns at once.

---

#### 12.7 — Final Check ✅
Print `X.shape`, `y.value_counts()`, and confirm zero NaNs. If everything looks clean — your data is ready for scikit-learn.

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
