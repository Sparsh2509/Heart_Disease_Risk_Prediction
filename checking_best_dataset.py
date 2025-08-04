import pandas as pd

# Load the datasets
upload_df = pd.read_csv("D:\Sparsh\ML_Projects\Heart_Disease_Prediction\Dataset\heart_cleveland_upload.csv")
new_df = pd.read_csv("D:\Sparsh\ML_Projects\Heart_Disease_Prediction\Dataset\Heart_disease_cleveland_new.csv")

# Convert 'condition' to binary 'target' in upload dataset
upload_df['target'] = upload_df['condition'].apply(lambda x: 1 if x > 0 else 0)


# Display target class distribution
print("Upload.csv Target Distribution:")
print(upload_df['target'].value_counts())

print("\nNew.csv Target Distribution:")
print(new_df['target'].value_counts())

# Check for missing values
print("\nMissing Values in Upload.csv:")
print(upload_df.isnull().sum())

print("\nMissing Values in New.csv:")
print(new_df.isnull().sum())
