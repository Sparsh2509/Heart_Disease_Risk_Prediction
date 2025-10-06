import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
df = pd.read_csv(r"D:\Sparsh\ML_Projects\Heart_Disease_Prediction\Dataset\Heart_disease_cleveland_new.csv")

# -------------------------------
# 2️⃣ Feature Engineering
# -------------------------------
# Keep logical, interpretable flags
df["fbs_flag"] = (df["fbs"] > 120).astype(int)
df["restecg_flag"] = (df["restecg"] != 0).astype(int)

# ✅ DO NOT ADD high_chol_flag (it created reverse pattern)
X = df.drop("target", axis=1)
y = df["target"]

# -------------------------------
# 3️⃣ Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 4️⃣ Standardization
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 5️⃣ Train Model (Random Forest)
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=2
)
model.fit(X_train_scaled, y_train)

# -------------------------------
# 6️⃣ Evaluate
# -------------------------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.3f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# 7️⃣ Save Model + Scaler
# -------------------------------
joblib.dump(model, "rf_heart_model.joblib")
joblib.dump(scaler, "rf_scaler.joblib")
print("\nModel and scaler saved successfully ✅")

# -------------------------------
# 8️⃣ Optional: Feature Importance
# -------------------------------
feat_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=feat_importance.values, y=feat_importance.index)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

# -------------------------------
# 9️⃣ Correlation Heatmap
# -------------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Heart Disease Dataset")
plt.show()
