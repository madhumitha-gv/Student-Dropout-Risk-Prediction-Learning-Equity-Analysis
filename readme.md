# Student Dropout Risk Prediction & Learning Equity Analysis

Predicting whether an online university student will withdraw or fail their course,
and analyzing equity gaps across demographic groups using the Open University
Learning Analytics Dataset (OULAD).

**Dataset:** [OULAD on Kaggle](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)

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
|-------|-------------------|-------|
| `0` — Success | Pass, Distinction | 18,302 |
| `1` — Dropout | Withdrawn, Fail | 22,435 |

**Class balance:** 55% dropout, 45% success

---

## Project Phases

1. Data Loading & Validation
2. Feature Engineering (VLE engagement, assessment behavior, demographics)
3. Data Cleaning & Encoding
4. XGBoost Dropout Prediction Model
5. SHAP Explainability
6. Equity Gap Analysis (disability, IMD deprivation, age)
