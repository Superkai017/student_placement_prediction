Student Placement Prediction AI
📌 Overview

This project builds a machine learning model to predict student placement outcomes based on academic, technical, practical, and soft skills. Feature engineering is applied to create meaningful skill-based indicators that improve prediction performance.

🎯 Objectives

Predict whether a student will be placed

Analyze key factors affecting placement

Create composite features (technical, practical, soft, academic skills)

Build and evaluate machine learning models

📂 Dataset

Synthetic student placement dataset

~100,000 rows, multiple features

Includes:

Academic: cgpa, backlogs

Technical: coding_skills, dsa_score, etc.

Practical: internships, projects, hackathons , open_source_contribution

Soft skills: communication_skills, extracurriculars

⚙️ Features Engineering

New features created:

technical_skill

practical_skill

soft_skill

academic_skill

MinMax scaling was used before aggregation.

🤖 Models Used

Logistic Regression

Linear regression

(Optional) XGBoost random forest

📈 Evaluation Metrics

Accuracy

Precision

F1-score

🛠️ Technologies

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

🚀 How to Run
pip install -r requirements.txt
python train.py
python predict.py
📁 Project Structure
project/
│── data/
│── notebooks/
│── models/
│── src/
│── train.py
│── predict.py
│── README.md
📌 Conclusion

This project demonstrates how machine learning and feature engineering can improve placement prediction and provide insights into student employability.
