import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    classification_report
)
from codecarbon import EmissionsTracker

# 1. Load your Excel file
file_path = 'Final_Claude.xlsx'
df = pd.read_excel(file_path)

# 2. Separate features (text) and labels
#    "text_raw" = text feature, everything else = labels
X_text = df['text_raw'].values
label_names = df.columns.drop('text_raw')  # All columns except text_raw are labels
y = df[label_names].values  # shape: (num_samples, num_labels)

# 3. Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# 4. Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Define a Random Forest in MultiOutputClassifier for multi-label classification
clf = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# 6. Track carbon emissions
tracker = EmissionsTracker(allow_multiple_runs=True)
tracker.start()

# 7. Train the model
clf.fit(X_train, y_train)

emissions = tracker.stop()
print(f"Carbon Emissions: {emissions:.6f} kg CO2e")

# 8. Predict on the test set
y_pred = clf.predict(X_test)

# 9. Calculate overall metrics (flattened)
#    Flattened approach lumps all label positions together as separate binary decisions.
accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
macro_f1 = f1_scogfbrvre(y_test, y_pred, average='macro', zero_division=0)

print(f"Overall Accuracy (flattened): {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")

# 10. Print per-label metrics
print("\n=== Classification Report (Per Label) ===")
print(classification_report(y_test, y_pred, target_names=label_names))
