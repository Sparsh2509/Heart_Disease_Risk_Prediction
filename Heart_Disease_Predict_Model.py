import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("D:\Sparsh\ML_Projects\Heart_Disease_Prediction\Dataset\Heart_disease_cleveland_new.csv")

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Evaluate
score = svm_model.score(X_test, y_test)
print(f"Accuracy Score: {score * 100:.2f}%")

# Save the model
joblib.dump(svm_model, "svm_heart_model.joblib")

# Optional: Save the scaler too if you plan to use it later
joblib.dump(scaler, "scaler.joblib")

print("Model and scaler saved")
