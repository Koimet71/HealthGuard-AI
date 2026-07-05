import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

print("✅ Dataset loaded")
print(df.iloc[:, 1:].values.flatten())

# -----------------------------
# STEP 1: DEFINE IMPORTANT SYMPTOMS
# -----------------------------
selected_symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "continuous_sneezing",
    "shivering",
    "chills",
    "joint_pain",
    "stomach_pain",
    "acidity",
    "ulcers_on_tongue",
    "muscle_wasting",
    "vomiting",
    "burning_micturition",
    "spotting_urination",
    "fatigue",
    "weight_gain",
    "anxiety",
    "cold_hands_and_feets",
    "mood_swings",
    "weight_loss"
]


# -----------------------------
# STEP 2: CONVERT TO BINARY
# -----------------------------
cleaned = pd.DataFrame()

cleaned = pd.DataFrame()

for symptom in selected_symptoms:
    cleaned[symptom] = df.iloc[:, 1:].apply(
        lambda row: int(
            symptom in [
                str(x).strip().lower() for x in row.values if pd.notna(x)
            ]
        ),
        axis=1
    )

print("✅ Symptoms converted to binary")

# -----------------------------
# STEP 3: MAP DISEASE → RISK
# -----------------------------
def map_risk(disease):
    disease = disease.lower()

    if disease in ["pneumonia", "malaria", "dengue", "typhoid"]:
        return "High"
    elif disease in ["common cold", "allergy", "flu"]:
        return "Medium"
    else:
        return "Low"

cleaned["risk"] = df["Disease"].apply(map_risk)

print("✅ Risk mapping done")

# -----------------------------
# STEP 4: SHOW RESULT
# -----------------------------
print(cleaned.head())

from sklearn.utils import resample

# Separate classes
low = cleaned[cleaned["risk"] == "Low"]
medium = cleaned[cleaned["risk"] == "Medium"]
high = cleaned[cleaned["risk"] == "High"]

# Upsample minority classes
medium_upsampled = resample(medium, replace=True, n_samples=len(low), random_state=42)
high_upsampled = resample(high, replace=True, n_samples=len(low), random_state=42)

# Combine
balanced = pd.concat([low, medium_upsampled, high_upsampled])

# Shuffle
balanced = balanced.sample(frac=1)
# -----------------------------
# STEP 5: TRAIN MODEL
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Features (X)
X = cleaned.drop("risk", axis=1)

y = df["Disease"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

print("✅ Disease prediction model ready!")

importances = model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))

plt.barh(
    importance_df['Feature'],
    importance_df['Importance']
)

plt.xlabel("Importance Score")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.show()
# -----------------------------
# STEP 6: TEST MODEL
# -----------------------------
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("📊 Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Random Forest Confusion Matrix")

plt.show()
print("📊 Confusion Matrix:")
print(cm)
print(cleaned["risk"].value_counts())