import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"D:\Sparsh\ML_Projects\Heart_Disease_Prediction\Dataset\Heart_disease_cleveland_new.csv")


# Features and target
X = df.drop("target", axis=1)  
y = df["target"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Train Random Forest Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=2
)
model.fit(X_train, y_train)


# Model prediction & evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.3f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Save model & feature names
joblib.dump(model, "randomforest_heart_model.joblib")
joblib.dump(X.columns.tolist(), "feature_columns.joblib")
print("\nModel and feature names saved successfully ✅")


# Feature Importance
feat_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=feat_importance.values, y=feat_importance.index)
plt.title("Feature Importance By Random Forest Algo")
plt.tight_layout()
plt.show()


# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Heart Disease Dataset")
plt.show()
