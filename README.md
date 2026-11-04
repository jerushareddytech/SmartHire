# SmartHire — Resume-to-Job Matching & Career Guidance Engine

SmartHire is a Machine Learning-based resume analysis and job recommendation system. It helps candidates identify suitable job opportunities based on their resumes, predict their professional role category, and understand which skills they need to improve.

The project uses classical Machine Learning and Natural Language Processing techniques without relying on Large Language Models (LLMs).

## 1. Project Overview

SmartHire allows users to upload their resumes and receive:
- Predicted resume/job role category.
- Top 5 relevant job recommendations.
- Resume skill extraction.
- Skill-gap analysis for selected job roles.

The main objective is to simplify the job-search process and provide useful career guidance through Machine Learning.

## 2. Core Features

### 2.1 Resume Category Classification

- Uses a supervised Machine Learning approach.
- TF-IDF is used to convert resume text into numerical features.
- Logistic Regression is used for classification.
- Predicts the professional category of the uploaded resume.
- Model performance is evaluated using accuracy and a confusion matrix.

### 2.2 Job Recommendation System

- Uses TF-IDF vectorization and cosine similarity.
- Compares the uploaded resume with available job descriptions.
- Calculates similarity scores.
- Displays the top 5 matching job opportunities.

### 2.3 Skill-Gap Analysis

- Extracts skills from the uploaded resume.
- Compares the extracted skills with the requirements of a selected job role.
- Identifies skills the candidate already possesses.
- Displays missing skills that the candidate can learn.
- Calculates a skill-match percentage.

The application supports skill-gap analysis for the following roles:
- Java Developer
- Python Developer
- Data Analyst
- Machine Learning Engineer
- Web Developer

## 3. Technologies Used

| Category | Technologies |
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Text Processing | TF-IDF |
| Similarity Measurement | Cosine Similarity |
| Classification Algorithm | Logistic Regression |
| Web Application | Streamlit |
| Resume Extraction | pdfplumber, python-docx |
| Model Saving | Joblib |
| Data Visualization | Matplotlib, Seaborn |

## 4. Project Requirements

- Python 3.10 or above
- Required Python libraries listed in `requirements.txt`
- VS Code or another Python IDE

## 5. Installation and Execution

### Step 1: Open the Project

Open the SmartHire project folder in VS Code.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

Ensure the required cleaned job dataset is available in the project's data folder.

The application uses the cleaned job dataset for job recommendations.

If the required dataset files are not available, follow the dataset preparation instructions in the project.

### Step 5: Run the Application

From the project root directory, execute:

```bash
python -m streamlit run app/streamlit_app.py
```

The SmartHire application will open in your default web browser.

## 6. How the Application Works

1. The user uploads a resume.
2. The application extracts text from the uploaded document.
3. The extracted text is preprocessed.
4. The trained classification model predicts the resume category.
5. The job recommendation system compares the resume with job descriptions.
6. The top 5 matching job recommendations are displayed.
7. The user selects a job role for skill-gap analysis.
8. The application identifies existing and missing skills and displays the skill-match percentage.

## 7. Project Structure

```text
SmartHire/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   ├── resume_classifier.pkl
│   └── resume_vectorizer.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_resume_classifier.ipynb
│   ├── 03_recommender.ipynb
│   ├── 04_clustering_topics.ipynb
│   └── 05_fit_predictor.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── parsing/
│   ├── config.py
│   └── evaluate.py
│
├── reports/
│
├── tests/
│
├── download_data.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 8. Model Evaluation

The resume classification model is evaluated using:
- Accuracy score
- Confusion matrix

The job recommendation system uses cosine similarity scores to rank job descriptions based on their relevance to the uploaded resume.

The skill-gap module calculates the percentage of required skills found in the resume for the selected job role.

## 9. Future Enhancements

Possible future improvements include:
- Resume-based salary prediction.
- Job clustering and topic modelling.
- Job-fit or candidate suitability prediction.
- Additional job categories and skill profiles.
- Improved job recommendation relevance.

These are potential enhancements and are separate from the three core features of the current application.

## 10. Conclusion

SmartHire demonstrates how Machine Learning and Natural Language Processing can be applied to resume classification, job recommendation, and career guidance.

By combining supervised learning, text similarity techniques, and skill-gap analysis, the system provides candidates with relevant job opportunities and insights into the skills they can develop for their desired careers.