import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load Dataset
df = pd.read_csv("loan_data.csv")

# Step 2: Encode categorical variables
df['Employment_Status'] = LabelEncoder().fit_transform(df['Employment_Status'])
df['Approval'] = LabelEncoder().fit_transform(df['Approval'])  # Approved = 1, Rejected = 0

# Step 3: Select Features and Target
X = df[['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio', 'Employment_Status']]
y = df['Approval']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save Model
joblib.dump(model, "loan_approval_model1.pkl")
