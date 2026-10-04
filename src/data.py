from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_data():
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=10, random_state=42
    )
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    return X_tr, X_cal, X_te, y_tr, y_cal, y_te
