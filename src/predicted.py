

import pandas as pd
import matplotlib.pyplot as plt

from Trained import get_models_and_scaler


# =============================================================================
# LOAD MODELS & SCALER FROM train.py
# =============================================================================

print("Loading models from train.py ...\n")
models_dict, scaler = get_models_and_scaler()
print("✓ Models ready!\n")


# =============================================================================
# ENTER STUDENT DETAILS
# =============================================================================

my_student = {
    "college_tier"    : 3,
    "salary_available": 0,
    "tech_skill"      : 0.32,
    "soft_skill"      : 0.15,
    "practical_skill" : 0.40,
    "cgpa_scaled"     : 0.50,
    "backlogs_scaled" : 0.0,
    "academic_points" : 0.40,
    "branch_CSE"      : 0,
    "branch_Chemical" : 0,
    "branch_ECE"      : 0,
    "branch_EE"       : 0,
    "branch_IT"       : 1,
    "branch_ME"       : 0,
}


# =============================================================================
# PREDICT
# =============================================================================

student_df     = pd.DataFrame([my_student])
student_scaled = scaler.transform(student_df)

print("📋 Student details you entered:")
print(student_df.to_string(index=False))

print("\n🤖 Predictions from each model:")
print("─" * 45)

for name, model in models_dict.items():

    prediction      = model.predict(student_scaled)[0]
    probability     = model.predict_proba(student_scaled)[0]
    prob_not_placed = probability[0] * 100
    prob_placed     = probability[1] * 100
    result          = "✅ PLACED" if prediction == 1 else "❌ NOT PLACED"

    print(f"\n  Model   : {name}")
    print(f"  Result  : {result}")
    print(f"  Confidence:")
    print(f"    → Placed     : {prob_placed:.1f}%")
    print(f"    → Not Placed : {prob_not_placed:.1f}%")

print("\n" + "─" * 45)

votes = sum(
    1 for model in models_dict.values()
    if model.predict(student_scaled)[0] == 1
)

print(f"\n📊 {votes}/3 models predict PLACED")

if   votes == 3: print("🎉 All models agree — very likely PLACED!")
elif votes == 2: print("👍 Majority says PLACED — good chances.")
elif votes == 1: print("😐 Only 1 model says PLACED — uncertain.")
else:            print("😟 All models agree — unlikely to be placed.")


# =============================================================================
# CONFIDENCE CHART
# =============================================================================

model_names      = list(models_dict.keys())
placed_probs     = [models_dict[n].predict_proba(student_scaled)[0][1] * 100
                    for n in model_names]
not_placed_probs = [100 - p for p in placed_probs]

fig, ax = plt.subplots(figsize=(9, 4))
x      = range(len(model_names))
bars1  = ax.bar(x, placed_probs,     label="Placed ✅",     color="#6baed6", width=0.4)
bars2  = ax.bar(x, not_placed_probs, label="Not Placed ❌", color="#e07b7b",
                width=0.4, bottom=placed_probs)

for bar, val in zip(bars1, placed_probs):
    if val > 5:
        ax.text(bar.get_x() + bar.get_width()/2, val/2,
                f"{val:.1f}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

for bar, val, base in zip(bars2, not_placed_probs, placed_probs):
    if val > 5:
        ax.text(bar.get_x() + bar.get_width()/2, base + val/2,
                f"{val:.1f}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

ax.set_xticks(list(x))
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel("Probability (%)")
ax.set_ylim(0, 110)
ax.set_title("Model Confidence", fontsize=13, fontweight="bold")
ax.legend(loc="upper right")
ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
plt.tight_layout()
plt.show()