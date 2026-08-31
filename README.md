# 🎓 Student Placement Prediction

> *Can a model predict a student's career fate before they even graduate?*

Landing a job after graduation depends on far more than grades alone — it's the intersection of technical ability, practical experience, and interpersonal skills. This project explores that intersection through machine learning.

Using a synthetic dataset of ~100,000 students, this project engineers composite skill profiles (technical, practical, soft, and academic) and trains classification models to predict whether a student is likely to be placed. The goal is not just prediction accuracy, but **interpretability** — understanding *which* skills actually move the needle on employability.

### Why It Matters

Universities, career centers, and students themselves often lack actionable, data-driven insight into placement readiness. A reliable prediction model can help:

- 🎯 **Students** identify skill gaps early and focus their efforts
- 🏫 **Institutions** benchmark their programs against placement outcomes
- 🔍 **Recruiters** surface high-potential candidates beyond GPA alone

---

## 📋 Table of Contents

- [Overview](#-student-placement-prediction)
- [Dataset](#-dataset)
- [Feature Engineering](#️-feature-engineering)
- [Models Used](#-models-used)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Structure](#-project-structure)
- [Technologies](#️-technologies)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Conclusion](#-conclusion)

---

## 📂 Dataset

The project uses a **synthetic student placement dataset** with approximately 100,000 rows, designed to mirror real-world placement data distributions.

| Category | Features |
|---|---|
| **Academic** | `cgpa`, `backlogs` |
| **Technical** | `coding_skills`, `dsa_score`, and related scores |
| **Practical** | `internships`, `projects`, `hackathons`, `open_source_contribution` |
| **Soft Skills** | `communication_skills`, `extracurriculars` |
| **Target** | `placement_status` (Placed / Not Placed) |

---

## ⚙️ Feature Engineering

Raw features were aggregated into four composite skill scores to reduce noise and improve model performance.

**New Engineered Features:**

| Feature | Description |
|---|---|
| `technical_skill` | Aggregated score from coding, DSA, and related technical metrics |
| `practical_skill` | Aggregated score from internships, projects, and open-source work |
| `soft_skill` | Aggregated score from communication and extracurricular activities |
| `academic_skill` | Aggregated score from CGPA and backlog status |

> **Note:** MinMax scaling was applied to each raw feature *before* aggregation to ensure fair weighting across different scales.

---

## 🤖 Models Used

| Model | Type | Notes |
|---|---|---|
| Logistic Regression | Classification | Baseline linear model |
| Linear Regression | Regression (adapted) | Used for comparison |
| XGBoost / Random Forest | Ensemble (optional) | Higher accuracy, non-linear |

---

## 📈 Evaluation Metrics

The models are evaluated using:

- **Accuracy** — overall correct predictions
- **Precision** — of predicted "placed" students, how many were actually placed
- **F1-Score** — harmonic mean of precision and recall, useful for imbalanced classes

---

## 📁 Project Structure

```
student_placement_prediction/
│
├── data/
│   └── student_data.csv          # Raw dataset
│
├── notebooks/
│   └── exploration.ipynb         # EDA and prototyping
│
├── models/
│   └── trained_model.pkl         # Saved model artifacts
│
├── src/
│   ├── feature_engineering.py    # Composite skill creation & scaling
│   ├── preprocessing.py          # Data cleaning and preparation
│   └── evaluation.py             # Metric computation utilities
│
├── train.py                      # Model training entry point
├── predict.py                    # Run predictions on new data
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🛠️ Technologies

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML models, scaling, evaluation |
| XGBoost | Gradient boosting (optional) |
| Matplotlib & Seaborn | Visualization |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Superkai017/student_placement_prediction.git
cd student_placement_prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python train.py
```

**4. Run predictions**
```bash
python predict.py
```

---

## 📊 Results

All models were evaluated on a held-out test set. The target accuracy range was **73–78%**.

| Model | Test Accuracy | In Target Range? |
|---|---|---|
| Logistic Regression | 74.70% | ✅ Yes |
| Random Forest | 74.44% | ✅ Yes |
| Gradient Boosting | 74.61% | ✅ Yes|
| **Overall** | **74.00%** | ✅ Yes |

**Key observations:**
- Logistic Regression achieved the highest accuracy despite being the simplest model — a strong signal that the engineered composite features created clean, linearly separable patterns
- Random Forest performed nearly identically, confirming the results are stable
- Gradient Boosting fell slightly short of the target range and could benefit from hyperparameter tuning (e.g., lower learning rate, more estimators)
- `technical_skill` and `practical_skill` were among the strongest predictors
- `cgpa` alone was not sufficient — composite features outperformed raw academic scores

---

## 📌 Conclusion

This project demonstrates how thoughtful **feature engineering** combined with machine learning can meaningfully improve placement prediction — and shed light on *what actually drives employability*. By aggregating raw signals into composite skill scores, the models gain a cleaner, more interpretable view of each student's profile.

Future improvements could include:
- Hyperparameter tuning with cross-validation
- SHAP values for feature explainability
- A simple web interface for real-time predictions

---

## 🙌 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).