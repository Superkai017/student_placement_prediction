import pandas as pd
import matplotlib.pyplot as plt
from Trained import get_models_and_scaler

print("Loading models ...\n")
models_dict, scaler = get_models_and_scaler()
print("✓ Models ready!\n")


# =============================================================================
# USER INPUT
# =============================================================================

print("=" * 45)
print("  STUDENT PLACEMENT PREDICTOR")
print("=" * 45)
print("Please enter the student details below.")
print("(Press Enter after each value)\n")

def get_float(prompt, min_val=0.0, max_val=1.0):
    while True:
        try:
            val = float(input(f"  {prompt} ({min_val}–{max_val}): "))
            if min_val <= val <= max_val:
                return val
            print(f" Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print(" Please enter a valid number")

def get_int(prompt, options):
    while True:
        try:
            val = int(input(f"  {prompt} ({'/'.join(map(str, options))}): "))
            if val in options:
                return val
            print(f"  Please choose from: {options}")
        except ValueError:
            print(" Please enter a valid number")

def get_branch():
    branches = ["CSE", "Chemical", "ECE", "EE", "IT", "ME"]
    print("\n  Branch options:")
    for i, b in enumerate(branches, 1):
        print(f"    {i}. {b}")
    while True:
        try:
            choice = int(input("  Choose branch number (1–6): "))
            if 1 <= choice <= 6:
                return branches[choice - 1]
            print(" Please choose between 1 and 6")
        except ValueError:
            print("  Please enter a valid number")

# ── Collect inputs ────────────────────────────────────────────
print("\n── College Info ──────────────────────────────")
college_tier = get_int("College tier (1=top, 2=average, 3=lower)", [1, 2, 3])

print("\n── Skills ────────────────────────────────────")
tech_skill      = get_float("Tech skill score      ")
soft_skill      = get_float("Soft skill score      ")
practical_skill = get_float("Practical skill score ")

print("\n── Academics ─────────────────────────────────")
cgpa_scaled     = get_float("CGPA scaled           ")
backlogs_scaled = get_float("Backlogs scaled       ")
academic_points = get_float("Academic points       ")

print("\n── Branch ────────────────────────────────────")
chosen_branch   = get_branch()

# ── Build student dict ────────────────────────────────────────
all_branches = ["CSE", "Chemical", "ECE", "EE", "IT", "ME"]
my_student = {
    "college_tier"    : college_tier,
    "tech_skill"      : tech_skill,
    "soft_skill"      : soft_skill,
    "practical_skill" : practical_skill,
    "cgpa_scaled"     : cgpa_scaled,
    "backlogs_scaled" : backlogs_scaled,
    "academic_points" : academic_points,
    "branch_CSE"      : 0,
    "branch_Chemical" : 0,
    "branch_ECE"      : 0,
    "branch_EE"       : 0,
    "branch_IT"       : 0,
    "branch_ME"       : 0,
}
my_student[f"branch_{chosen_branch}"] = 1


# =============================================================================
# PREDICT
# =============================================================================

student_df     = pd.DataFrame([my_student])
student_scaled = scaler.transform(student_df)

print("\n" + "=" * 45)
print("  STUDENT PROFILE SUMMARY")
print("=" * 45)
print(f"  College Tier    : {college_tier}  (1=top, 2=avg, 3=lower)")
print(f"  Tech Skill      : {tech_skill}")
print(f"  Soft Skill      : {soft_skill}")
print(f"  Practical Skill : {practical_skill}")
print(f"  CGPA Scaled     : {cgpa_scaled}")
print(f"  Backlogs        : {backlogs_scaled}")
print(f"  Academic Points : {academic_points}")
print(f"  Branch          : {chosen_branch}")

print("\n🤖 Predictions from each model:")
print("─" * 45)

THRESHOLD = 0.70
for name, model in models_dict.items():
    probability     = model.predict_proba(student_scaled)[0]
    prob_not_placed = probability[0] * 100
    prob_placed     = probability[1] * 100
    prediction      = 1 if prob_placed / 100 >= THRESHOLD else 0
    result          = "PLACED" if prediction == 1 else " NOT PLACED"

    print(f"\n  Model   : {name}")
    print(f"  Result  : {result}")
    print(f"  Confidence:")
    print(f"    → Placed     : {prob_placed:.1f}%")
    print(f"    → Not Placed : {prob_not_placed:.1f}%")

print("\n" + "─" * 45)

votes = sum(
    1 for model in models_dict.values()
    if model.predict_proba(student_scaled)[0][1] >= THRESHOLD
)

print(f"\n {votes}/3 models predict PLACED")

if   votes == 3: print(" based on majority vote — very likely PLACED!")
elif votes == 2: print(" based on majority vote — good chances to be placed but need more assessment.")
else:            print(" based on majority vote — unlikely to be placed.")


# =============================================================================
# CONFIDENCE CHART
# =============================================================================

model_names      = list(models_dict.keys())
placed_probs     = [models_dict[n].predict_proba(student_scaled)[0][1] * 100
                    for n in model_names]
not_placed_probs = [100 - p for p in placed_probs]

fig, ax = plt.subplots(figsize=(9, 4))
x      = range(len(model_names))
bars1  = ax.bar(x, placed_probs,     label="Placed",     color="#6baed6", width=0.4)
bars2  = ax.bar(x, not_placed_probs, label="Not Placed", color="#e07b7b",
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
ax.set_title(f"Model Confidence — {chosen_branch} | Tier {college_tier} | CGPA {cgpa_scaled}",
             fontsize=12, fontweight="bold")

# ── Threshold line at 70% ─────────────────────────────────────
ax.axhline(70, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
ax.text(2.55, 71.5, "Threshold 70%", color="red", fontsize=10, fontweight="bold")

ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
votes_placed = sum(
    1 for n, m in models_dict.items()
    if m.predict_proba(student_scaled)[0][1] >= THRESHOLD
)

print(f"\n Final Verdict: {votes_placed} out of 3 models predict PLACED ")

if votes_placed == 3:
    print("All models agree — this student is very likely to get PLACED!")
elif votes_placed == 2:
    print("Majority says PLACED — good chances for this student.")
else:
    print("All models agree — this student is unlikely to be placed.")