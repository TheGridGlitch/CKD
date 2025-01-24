import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import joblib

file_path = 'synthetic_kidney_disease_data.csv'

# Load dataset with appropriate encoding
try:
    data = pd.read_csv(file_path, encoding='latin1')
    print("Dataset loaded successfully with 'latin1' encoding.")
except UnicodeDecodeError:
    print("The file has a different encoding. Trying 'cp1252'.")
    data = pd.read_csv(file_path, encoding='cp1252')
    print("Dataset loaded successfully with 'cp1252' encoding.")

# Drop the 'id' column if it exists
if 'id' in data.columns:
    data = data.drop(columns=['id'])

# Handle missing values
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].mean())

# One-hot encode categorical features
data = pd.get_dummies(data, drop_first=True)

# Define features and target
if 'classification_notckd' in data.columns:
    X = data.drop('classification_notckd', axis=1)
    y = data['classification_notckd']
else:
    raise ValueError("Target column 'classification_notckd' not found in the dataset.")

# **Step 1: Save the feature names for later use**
feature_names_filename = "training_feature_names.pkl"
joblib.dump(X.columns.tolist(), feature_names_filename)
print(f"Feature names saved as {feature_names_filename}.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLflow experiment
mlflow.set_experiment("Kidney Disease Classification")

with mlflow.start_run():
    # Train the model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Log model and metrics to MLflow
    input_example = X_test.iloc[0:1].to_dict(orient='records')  # First row of the test set as an example
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        signature=mlflow.models.infer_signature(X_test, y_pred)
    )
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    print("Model and metrics logged to MLflow.")

    # Save the model as a .pkl file for UI
    model_filename = "logistic_regression_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}.")

