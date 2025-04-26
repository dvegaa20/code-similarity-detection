from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def train_model(X, y):
    """
    Trains a Support Vector Classifier model using the given features and labels.

    Parameters
    ----------
    X : array-like
        The input features for the model.
    y : array-like
        The target labels for the model.

    Returns
    -------
    model : SVC
        The trained Support Vector Classifier model.
    X_test : array-like
        The features for the test dataset.
    y_test : array-like
        The labels for the test dataset.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the given model on the test dataset.

    Parameters
    ----------
    model : SVC
        The Support Vector Classifier model to evaluate.
    X_test : array-like
        The test dataset features.
    y_test : array-like
        The test dataset labels.

    Returns
    -------
    report : dict
        A dictionary with the classification report.
    """
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report
