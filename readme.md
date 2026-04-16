# Student Dropout Risk Prediction & Learning Equity Analysis

Predicting whether an online university student will withdraw or fail their course,
and analyzing equity gaps across demographic groups using the Open University
Learning Analytics Dataset (OULAD).

**Dataset:** [OULAD on Kaggle](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)  
**Model Performance:** 92% Accuracy | 0.976 ROC-AUC

---

## Problem Statement

Online universities face a critical challenge: students who are at risk of dropping
out or failing are often not identified until it is too late to intervene. This
project builds an early warning system that predicts dropout risk from student
engagement, assessment behavior, and demographic data — enabling educators to
intervene early and improve outcomes.

A secondary goal is equity analysis: identifying whether certain demographic groups
(students with disabilities, students from deprived areas, older students) face
systematically higher dropout rates, and quantifying those gaps to support
data-informed curriculum intervention strategies.

---

## About the Dataset

The Open University is the largest academic institution in the UK with 2 million
enrolled students since 1969. This dataset captures student interactions with its
Virtual Learning Environment (VLE) — an online platform used for course content,
assessments, forum discussions, and assignment feedback — across 7 selected courses
and multiple semester presentations.

**Source:** Kuzilek J., Hlosta M., Zdrahal Z. *Open University Learning Analytics
dataset* Sci. Data 4:170171 (2017). Licensed under CC BY 4.0.

---

## Dataset Files (7 CSVs)

### 1. `studentInfo.csv` — 32,593 rows
The core demographic table. One row per student per module-presentation enrollment.

| Column | Description |
|--------|-------------|
| `code_module` | Course identifier (AAA, BBB, etc.) |
| `code_presentation` | Semester (e.g. 2013J = 2013 Semester 1) |
| `id_student` | Unique student ID |
| `gender` | M / F |
| `region` | UK region of the student |
| `highest_education` | Highest prior education level |
| `imd_band` | Index of Multiple Deprivation — socioeconomic indicator (0–100%) |
| `age_band` | Age group (0-35, 35-55, 55<=) |
| `num_of_prev_attempts` | How many times student previously attempted this module |
| `studied_credits` | Total credits student is studying |
| `disability` | Whether student has a declared disability (Y/N) |
| `final_result` | **Target**: Pass, Distinction, Fail, or Withdrawn |

---

### 2. `studentVle.csv` — 10,655,280 rows
Every click interaction a student made on the VLE platform. The largest file.

| Column | Description |
|--------|-------------|
| `code_module` | Course identifier |
| `code_presentation` | Semester |
| `id_student` | Student ID |
| `id_site` | VLE activity/page ID |
| `date` | Days since course start (day 0 = start) |
| `sum_click` | Number of clicks on that day for that activity |

---

### 3. `vle.csv` — 6,364 rows
Metadata about each VLE activity/page.

| Column | Description |
|--------|-------------|
| `id_site` | VLE activity ID |
| `code_module` | Course identifier |
| `code_presentation` | Semester |
| `activity_type` | Type of resource (forum, quiz, resource, homepage, etc.) |
| `week_from` | Week the activity becomes available |
| `week_to` | Week the activity is no longer active |

---

### 4. `studentAssessment.csv` — 173,912 rows
Each student's submission record for every assessment they attempted.

| Column | Description |
|--------|-------------|
| `id_assessment` | Assessment ID |
| `id_student` | Student ID |
| `date_submitted` | Day the student submitted (days since course start) |
| `is_banked` | Whether the score was carried over from a previous attempt |
| `score` | Score achieved (0–100) |

---

### 5. `assessments.csv` — 206 rows
Metadata about each assessment.

| Column | Description |
|--------|-------------|
| `code_module` | Course identifier |
| `code_presentation` | Semester |
| `id_assessment` | Assessment ID |
| `assessment_type` | TMA (Tutor Marked), CMA (Computer Marked), or Exam |
| `date` | Due date in days since course start |
| `weight` | Assessment weight in % (Exams = 100%, others sum to 100%) |

---

### 6. `studentRegistration.csv` — 32,593 rows
Enrollment and withdrawal dates per student per module.

| Column | Description |
|--------|-------------|
| `code_module` | Course identifier |
| `code_presentation` | Semester |
| `id_student` | Student ID |
| `date_registration` | Day student registered (relative to course start) |
| `date_unregistration` | Day student unregistered — NULL if they didn't withdraw |

---

### 7. `courses.csv` — 22 rows
Basic metadata about each course offering.

| Column | Description |
|--------|-------------|
| `code_module` | Course identifier |
| `code_presentation` | Semester |
| `module_presentation_length` | Total length of the course in days |

---

## Target Variable

| Label | final_result values | Count |
|-------|---------------------|-------|
| `0` — Success | Pass, Distinction | 18,302 |
| `1` — Dropout | Withdrawn, Fail | 22,435 |

**Class balance:** 55.07% dropout, 44.93% success — mild imbalance handled via
`scale_pos_weight` in XGBoost.

---

## Project Phases

### Phase 1 — Data Loading & Validation

All 7 CSVs were loaded into pandas DataFrames and validated for shape and column
structure. Confirmed 32,593 student records, 10,655,280 VLE interactions, and all
expected join keys present across datasets. The student-module level grain (40,737
rows after cleaning) was chosen over unique student level to preserve repeat attempt
information and maximize training signal.

---

### Phase 2 — Feature Engineering

Three feature groups were engineered and merged onto the base student demographics
table, producing a final feature matrix of 40,737 rows × 52 features.

**VLE Engagement Features** (aggregated from 10M+ rows in studentVle.csv):

| Feature | Description |
|---------|-------------|
| `total_clicks` | Total platform interactions across the course |
| `active_days` | Number of distinct days the student logged in |
| `avg_clicks_per_day` | Average engagement intensity per active day |
| `max_clicks_day` | Peak single-day engagement |
| `early_clicks` | Total clicks in first 30 days — early engagement signal |

**Assessment Behavior Features** (studentAssessment.csv joined with assessments.csv):

| Feature | Description |
|---------|-------------|
| `avg_score` | Mean score across all assessments attempted |
| `min_score` | Worst single assessment score |
| `num_assessments` | Total assessments attempted |
| `avg_days_early` | Average days before deadline submissions were made (negative = late) |
| `num_late_submissions` | Count of assessments submitted after the deadline |

**Registration Features** (studentRegistration.csv):

| Feature | Description |
|---------|-------------|
| `date_registration` | How early or late the student registered relative to course start |
| `unregistered` | Binary flag for whether the student formally withdrew |

Demographics from `studentInfo.csv` retained directly: `gender`, `region`,
`highest_education`, `imd_band`, `age_band`, `disability`, `num_of_prev_attempts`,
`studied_credits`.

---

### Phase 3 — Data Cleaning & Encoding

| Issue | Count | Resolution |
|-------|-------|------------|
| VLE nulls (no platform activity) | 3,152 | Filled with 0 — absence of engagement is itself a dropout signal |
| Assessment nulls (no submissions) | 6,794 | Filled with 0 — no submissions is a strong risk indicator |
| IMD band nulls | 1,401 | Filled with mode — conservative imputation for missing demographic data |
| Missing registration dates | 64 | Rows dropped |

Categorical columns encoded with `pd.get_dummies` (one-hot encoding). Column names
sanitized to remove brackets and special characters incompatible with XGBoost.
**Zero nulls remaining after cleaning.**

---

### Phase 4 — XGBoost Model Training & Evaluation

**Train/Test Split:** 80/20 stratified split → 32,589 train | 8,148 test

**Class imbalance:** Handled via `scale_pos_weight = 0.816` — no oversampling needed.

**Model configuration:**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 300 (with early stopping, patience 20) |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `eval_metric` | AUC |

**Results:**

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.9760** |
| Accuracy | **92%** |
| Precision (Dropout) | 96% |
| Recall (Dropout) | 89% |
| F1-Score (Dropout) | 93% |

The model converged steadily from AUC 0.929 at round 0 to 0.976 at round 299 with
no signs of overfitting. High precision (96%) means very few false alarms — students
flagged as at-risk are almost always genuinely at risk. Recall of 89% means the
model catches the vast majority of students who will drop out.

---

### Phase 5 — SHAP Explainability

SHAP (SHapley Additive exPlanations) values were computed using `TreeExplainer`
to identify which features drive dropout predictions globally and per student.

**Top 10 Dropout Risk Drivers:**

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|------|---------|--------------|----------------|
| 1 | `num_assessments` | 2.184 | Fewer assessments attempted = highest dropout risk |
| 2 | `active_days` | 0.774 | Low platform login frequency strongly predicts dropout |
| 3 | `avg_score` | 0.757 | Low average scores push toward dropout |
| 4 | `unregistered` | 0.577 | Formal withdrawal is a near-certain dropout signal |
| 5 | `early_clicks` | 0.506 | Low engagement in first 30 days — most actionable for early intervention |
| 6 | `code_module_CCC` | 0.410 | Module CCC has elevated risk independent of student behavior |
| 7 | `avg_days_early` | 0.295 | Consistently late submissions signal higher risk |
| 8 | `total_clicks` | 0.286 | Overall platform engagement matters beyond just login days |
| 9 | `min_score` | 0.253 | A single very poor assessment score is a warning sign |
| 10 | `code_presentation_2014J` | 0.238 | Semester-level effects suggest cohort or curriculum variation |

> **Key insight #1:** `num_assessments` is the dominant signal — nearly 3x stronger
> than any other feature. Students disengaging from assessments are at far greater
> risk than students scoring poorly. Interventions should focus on getting students
> to *attempt* assessments, not just improve scores.

> **Key insight #2:** `early_clicks` is the most *actionable* feature — low
> engagement in the first 30 days is detectable before dropout occurs, creating
> a clear window for early outreach and intervention.

---

### Phase 6 — Equity Gap Analysis

Chi-square hypothesis testing was conducted across 4 demographic dimensions to
identify statistically significant disparities in dropout rates. All 4 showed
highly significant gaps (p < 0.0001).

**Disability Status**

| Group | Dropout Rate | Count |
|-------|-------------|-------|
| No disability | 54.0% | 36,582 |
| Disability declared | 64.4% | 4,155 |

**Gap: +10.4pp** | Chi-square: 163.26 | p < 0.0001

Students with declared disabilities drop out at a 10.4pp higher rate, suggesting
existing support structures are insufficient for disabled learners.

---

**IMD Deprivation Band** (0% = most deprived, 100% = least deprived)

| Band | Dropout Rate |
|------|-------------|
| 0–10% (most deprived) | 66.5% |
| 10–20% | 63.2% |
| 20–30% | 55.9% |
| 30–40% | 55.1% |
| 40–50% | 55.9% |
| 50–60% | 53.2% |
| 60–70% | 50.6% |
| 70–80% | 51.5% |
| 80–90% | 48.8% |
| 90–100% (least deprived) | 45.2% |

**Gap: +21.3pp** (most vs least deprived) | Chi-square: 567.96 | p < 0.0001

A clear socioeconomic gradient exists across all 10 bands. Students from the most
deprived areas drop out at nearly 1.5x the rate of least deprived students.

---

**Age Group**

| Group | Dropout Rate | Count |
|-------|-------------|-------|
| 0–35 | 57.0% | 28,933 |
| 35–55 | 50.7% | 11,508 |
| 55+ | 41.6% | 296 |

**Gap: +15.4pp** (youngest vs oldest) | Chi-square: 154.84 | p < 0.0001

Older students perform significantly better, likely due to higher intrinsic
motivation. Younger students may benefit from additional goal-setting support.

---

**Highest Education Level**

| Education Level | Dropout Rate | Count |
|----------------|-------------|-------|
| No formal qualifications | 72.5% | 433 |
| Lower than A Level | 63.2% | 16,099 |
| A Level or Equivalent | 50.7% | 17,671 |
| HE Qualification | 46.3% | 6,081 |
| Post Graduate Qualification | 38.6% | 453 |

**Gap: +33.9pp** (no quals vs postgrad) | Chi-square: 854.24 | p < 0.0001

The largest equity gap in the dataset. Students with no formal qualifications drop
out at nearly double the rate of postgraduates, highlighting the need for
foundation-level support for underprepared learners.

---

## Results Summary

### Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9760 |
| Accuracy | 92% |
| Precision (Dropout) | 96% |
| Recall (Dropout) | 89% |
| F1-Score (Dropout) | 93% |

### Equity Gaps

| Dimension | High Risk Group | Low Risk Group | Gap |
|-----------|----------------|----------------|-----|
| Disability | 64.4% (Y) | 54.0% (N) | +10.4pp |
| Deprivation | 66.5% (0–10%) | 45.2% (90–100%) | +21.3pp |
| Age | 57.0% (0–35) | 41.6% (55+) | +15.4pp |
| Education | 72.5% (No quals) | 38.6% (PostGrad) | +33.9pp |

All gaps statistically significant at p < 0.0001 via chi-square test.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Languages | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, scikit-learn |
| Explainability | SHAP |
| Statistical Testing | SciPy (chi-square) |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebooks |
