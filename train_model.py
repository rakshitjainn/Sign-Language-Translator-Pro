import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

try:
    data = pd.read_csv('hand_data.csv')
    print(f"Data Loaded Successfully! Found {len(data)} samples.")
except FileNotFoundError:
    print("Error: 'hand_data.csv' not found. Please run collect_data.py first.")
    exit()

X = data.drop('class_id', axis=1) 
y = data['class_id']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- TRAINING STARTED ---")

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred) * 100
print(f"Random Forest Accuracy: {rf_acc:.2f}%")

print("Training Neural Network...")
nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)
nn_acc = accuracy_score(y_test, nn_pred) * 100
print(f"Neural Network Accuracy: {nn_acc:.2f}%")

print("\n--- RESULT ---")
if rf_acc >= nn_acc:
    print(f"Random Forest won! Saving model...")
    with open('sign_language_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
else:
    print(f"Neural Network won! Saving model...")
    with open('sign_language_model.pkl', 'wb') as f:
        pickle.dump(nn_model, f)

print("Model saved as 'sign_language_model.pkl'")