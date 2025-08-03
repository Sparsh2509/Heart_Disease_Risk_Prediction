import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("D:\Sparsh\ML_Projects\Heart_Disease_Prediction\Dataset\Heart_disease_cleveland_new.csv")


# High Cholesterol Flag (> 240 mg/dl)
df['high_chol_flag'] = 0
for i in range(len(df)):
    if df.loc[i, 'chol'] > 240:
        df.loc[i, 'high_chol_flag'] = 1

# High Fasting Blood Sugar Flag (fbs == 1)
df['fbs_flag'] = 0
for i in range(len(df)):
    if df.loc[i, 'fbs'] == 1:
        df.loc[i, 'fbs_flag'] = 1

# Abnormal RestECG Flag (restecg != 0)
df['restecg_flag'] = 0
for i in range(len(df)):
    if df.loc[i, 'restecg'] != 0:
        df.loc[i, 'restecg_flag'] = 1


X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and scaler
joblib.dump(knn, "knn_heart_model.joblib")
joblib.dump(scaler, "knn_scaler.joblib")

print("Model and scaler saved")

# Confusion Matrix Plot
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Heatmap of Heart Disease Dataset")
plt.tight_layout()
plt.show()
