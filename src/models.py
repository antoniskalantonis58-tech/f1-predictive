from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def baseline_predict(grid):
    return (grid <= 10).astype(int)

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
