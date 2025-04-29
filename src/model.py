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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    automl = AutoML()
    automl.fit(
        X_train,
        y_train,
        task="classification",
        time_budget=3600,
        metric="f1",
        log_file_name="flaml.log",
    )

    return automl, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model on a test dataset and returns a classification report.

    Parameters
    ----------
    model : object
        The trained model to be evaluated.
    X_test : array-like or DataFrame
        The input samples for testing.
    y_test : array-like
        The true labels for `X_test`.

    Returns
    -------
    dict
        A dictionary containing the classification report, including precision, recall,
        f1-score, and support for each class.
    """

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report
