from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from flaml import AutoML

def train_model(X, y):
    """
    Trains a model using FLAML.

    Returns
    -------
    model : trained AutoML model
    X_test, y_test : hold-out test set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    automl = AutoML()
    automl.fit(
        X_train, y_train,
        task="classification",
        time_budget=3600, # 60 minutes search
        metric='f1', # F1 score as optimization metric
        log_file_name="flaml.log"
    )

    return automl, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the FLAML model.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report
