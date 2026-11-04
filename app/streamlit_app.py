"""SmartHire web portal. Run: python -m streamlit run app/streamlit_app.py"""

import streamlit as st
import pandas as pd
import re
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

MODEL_DIR = PROJECT_DIR / "models"
CLASSIFIER_PATH = MODEL_DIR / "resume_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "resume_vectorizer.pkl"

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="SmartHire",
    page_icon="💼",
    layout="wide"
)

st.title("💼 SmartHire")
st.subheader("Resume-to-Job Matching & Career Guidance")

st.write(
    "Upload your resume to discover relevant job opportunities, "
    "predict your resume category, and explore career paths."
)

st.divider()


# --------------------------------------------------
# EXTRACT TEXT FROM RESUME
# --------------------------------------------------
def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="ignore"
        )

    elif filename.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(uploaded_file)

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    elif filename.endswith(".docx"):
        from docx import Document

        document = Document(uploaded_file)

        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    return ""


# --------------------------------------------------
# CLEAN RESUME TEXT
# --------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)

    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------
# LOAD JOB DATASET
# --------------------------------------------------
@st.cache_data
def load_jobs():
    possible_paths = [
        DATA_DIR / "jobs_clean.csv",
        PROJECT_DIR / "data" / "jobs_clean.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    return None


# --------------------------------------------------
# LOAD SAVED RESUME CLASSIFICATION MODEL
# --------------------------------------------------
@st.cache_resource
def load_resume_model():
    if CLASSIFIER_PATH.exists() and VECTORIZER_PATH.exists():
        classifier = joblib.load(CLASSIFIER_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        return classifier, vectorizer

    return None, None


# --------------------------------------------------
# SKILL KEYWORDS
# --------------------------------------------------
SKILL_KEYWORDS = [
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "machine learning",
    "data analysis",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "communication",
    "problem solving",
    "git",
    "linux",
    "aws",
    "excel",
    "tensorflow",
    "pytorch",
    "matlab",
    "deep learning",
    "data structures",
    "computer vision",
    "scikit-learn"
]


# --------------------------------------------------
# REQUIRED SKILLS FOR EACH JOB
# --------------------------------------------------
REQUIRED_SKILLS = {
    "Java Developer": {
        "java",
        "sql",
        "git",
        "problem solving",
        "communication"
    },

    "Python Developer": {
        "python",
        "sql",
        "git",
        "problem solving",
        "communication"
    },

    "Data Analyst": {
        "python",
        "sql",
        "excel",
        "data analysis",
        "communication"
    },

    "Machine Learning Engineer": {
        "python",
        "machine learning",
        "sql",
        "tensorflow",
        "pytorch",
        "problem solving"
    },

    "Web Developer": {
        "html",
        "css",
        "javascript",
        "react",
        "git"
    }
}


# --------------------------------------------------
# EXTRACT SKILLS FROM RESUME
# --------------------------------------------------
def extract_skills(resume_text):
    normalized_text = clean_text(resume_text)

    detected_skills = {
        skill
        for skill in SKILL_KEYWORDS
        if re.search(
            r"(?<!\w)" + re.escape(skill) + r"(?!\w)",
            normalized_text
        )
    }

    return detected_skills


# --------------------------------------------------
# RESUME UPLOAD
# --------------------------------------------------
st.markdown("### 📄 Upload Your Resume")

uploaded_file = st.file_uploader(
    "Choose a PDF, DOCX, or TXT resume",
    type=["pdf", "docx", "txt"]
)


if uploaded_file is not None:

    if st.button("Analyze Resume", type="primary"):

        try:
            # Extract and clean resume text
            resume_text = extract_text(uploaded_file)
            cleaned_resume = clean_text(resume_text)

            # Save resume text for use in other sections
            if cleaned_resume:
                st.session_state["resume_text"] = resume_text
                st.session_state["cleaned_resume"] = cleaned_resume
            else:
                st.session_state.pop("resume_text", None)
                st.session_state.pop("cleaned_resume", None)

            if not cleaned_resume:

                st.error(
                    "No readable text found. Please upload a "
                    "text-based PDF, DOCX, or TXT resume."
                )

            else:

                st.success("Resume uploaded successfully!")

                # ----------------------------------------------
                # RESUME PREVIEW
                # ----------------------------------------------
                st.markdown("### 📝 Extracted Resume Text")

                st.text_area(
                    "Resume preview",
                    value=resume_text[:3000],
                    height=180
                )

                # ----------------------------------------------
                # RESUME CATEGORY PREDICTION
                # ----------------------------------------------
                st.markdown("### 🧠 Resume Category Prediction")

                classifier, resume_vectorizer = load_resume_model()

                if classifier is not None and resume_vectorizer is not None:

                    try:
                        resume_features = resume_vectorizer.transform(
                            [cleaned_resume]
                        )

                        predicted_category = classifier.predict(
                            resume_features
                        )[0]

                        st.success(
                            f"Predicted Resume Category: "
                            f"**{predicted_category}**"
                        )

                    except Exception as model_error:

                        st.warning(
                            "Could not predict the resume category. "
                            f"Details: {model_error}"
                        )

                else:

                    st.warning(
                        "The trained classification model was not found. "
                        "Please run the model-saving cell in "
                        "02_resume_classifier.ipynb first."
                    )

                # ----------------------------------------------
                # LOAD JOB DATASET
                # ----------------------------------------------
                jobs = load_jobs()

                if jobs is None:

                    st.error(
                        "Job dataset not found. Check that "
                        "jobs_clean.csv exists in the data folder."
                    )

                elif "text" not in jobs.columns:

                    st.error(
                        "The job dataset does not contain a 'text' column."
                    )

                else:

                    jobs = jobs.copy()

                    jobs["text"] = (
                        jobs["text"]
                        .fillna("")
                        .astype(str)
                    )

                    jobs = jobs[
                        jobs["text"].str.strip() != ""
                    ]

                    if jobs.empty:

                        st.error(
                            "The job dataset contains no job descriptions."
                        )

                    else:

                        # ------------------------------------------
                        # JOB RECOMMENDATIONS
                        # ------------------------------------------
                        st.markdown("### 🎯 Top 5 Recommended Jobs")

                        with st.spinner("Finding matching jobs..."):

                            vectorizer = TfidfVectorizer(
                                max_features=10000,
                                stop_words="english"
                            )

                            job_matrix = vectorizer.fit_transform(
                                jobs["text"]
                            )

                            resume_vector = vectorizer.transform(
                                [cleaned_resume]
                            )

                            scores = cosine_similarity(
                                resume_vector,
                                job_matrix
                            ).flatten()

                            number_of_results = min(5, len(jobs))

                            top_indices = scores.argsort()[
                                -number_of_results:
                            ][::-1]

                            results = jobs.iloc[top_indices].copy()

                            results["Match Score (%)"] = (
                                scores[top_indices] * 100
                            ).round(2)

                            results = results.rename(
                                columns={
                                    "title": "Job Title",
                                    "company": "Company"
                                }
                            )

                            display_columns = [
                                column
                                for column in [
                                    "Job Title",
                                    "Company",
                                    "Match Score (%)"
                                ]
                                if column in results.columns
                            ]

                            display_results = (
                                results[display_columns]
                                .reset_index(drop=True)
                            )

                            st.dataframe(
                                display_results,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Job Title":
                                        st.column_config.TextColumn(
                                            "Job Title",
                                            width="large"
                                        ),

                                    "Company":
                                        st.column_config.TextColumn(
                                            "Company",
                                            width="medium"
                                        ),

                                    "Match Score (%)":
                                        st.column_config.ProgressColumn(
                                            "Match Score",
                                            min_value=0,
                                            max_value=100,
                                            format="%.2f%%"
                                        )
                                }
                            )

                        st.caption(
                            "Match scores represent text similarity, "
                            "not a guarantee of job suitability."
                        )

        except Exception as error:

            st.error(f"Something went wrong: {error}")

else:

    st.info("Upload your resume to get started.")


# --------------------------------------------------
# SKILL-GAP REPORT
# --------------------------------------------------
st.divider()

st.markdown("### 🛠️ Skill-Gap Analysis")

st.write(
    "Compare your detected resume skills with the skills required "
    "for your desired job."
)

# Select the desired job
desired_job = st.selectbox(
    "Choose a job to analyze",
    list(REQUIRED_SKILLS.keys())
)


if st.button("Generate Skill-Gap Report"):

    # Retrieve resume saved during Analyze Resume
    current_resume = st.session_state.get(
        "cleaned_resume",
        ""
    )

    if not current_resume:

        st.warning(
            "Please upload your resume and click Analyze Resume first."
        )

    else:

        # Extract skills from the actual uploaded resume
        resume_skills = extract_skills(current_resume)

        # Get required skills for selected job
        selected_required_skills = REQUIRED_SKILLS[desired_job]

        # Calculate matched and missing skills
        matched_skills = resume_skills.intersection(
            selected_required_skills
        )

        missing_skills = selected_required_skills.difference(
            resume_skills
        )

        # Calculate skill match percentage
        total_required = len(selected_required_skills)

        match_percentage = (
            len(matched_skills) / total_required * 100
            if total_required
            else 0
        )

        # Display match percentage
        st.metric(
            "Skill Match",
            f"{match_percentage:.1f}%"
        )

        # Display detected matching skills
        st.markdown("#### ✅ Skills You Have")

        if matched_skills:

            st.write(
                ", ".join(sorted(matched_skills))
            )

        else:

            st.info(
                "No matching skills from this job profile "
                "were detected in your resume."
            )

        # Display missing skills
        st.markdown("#### 📚 Skills to Learn")

        if missing_skills:

            st.write(
                ", ".join(sorted(missing_skills))
            )

        else:

            st.success(
                "All listed skills for this job were detected!"
            )

        st.caption(
            "This report uses keyword-based skill extraction. "
            "It may miss skills written differently and does not "
            "measure your proficiency."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption("SmartHire | Career Guidance System")