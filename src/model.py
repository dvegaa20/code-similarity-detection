from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

def train_model(X, y, kernel='linear', test_size=0.3, random_state=42):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train SVM
    model = SVC(kernel=kernel, probability=True, random_state=random_state)
    model.fit(X_train, y_train)

    return model, X_test, y_test, scaler

def evaluate_model(model, X_test, y_test, display=True):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    if display:
        print(classification_report(y_test, y_pred))
    
    return report

def cross_validate_model(X, y, kernel='linear', cv=5, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVC(kernel=kernel, random_state=random_state)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_macro')
    
    print(f"Cross-validated F1 macro scores: {scores}")
    print(f"Mean F1 macro: {np.mean(scores):.4f}")
    return scores
