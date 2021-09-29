import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def random_forest(df: pd.DataFrame, target: str, encoding: bool = False):
    X = df.drop(columns = [target])
    y = df[target].values.reshape(-1, 1)
    if encoding:
        encoder = OneHotEncoder()
        X = encoder.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators= 5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Train score : ", model.score(X_train, y_train))
    print("Test score : ", metrics.accuracy_score(y_test, y_pred))

def decision_tree(df: pd.DataFrame, target: str, encoding: bool = False, max_depth = None, score: str = "print"):
    X = df.drop(columns = [target])
    y = df[target].values.reshape(-1, 1)
    if encoding:
        encoder = OneHotEncoder()
        X = encoder.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

    model = DecisionTreeClassifier(max_depth= max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = metrics.accuracy_score(y_test, y_pred)
    if score == "print":
        print("Train score : ", train_score)
        print("Test score : ", test_score)
    if score == "return":
        return train_score, test_score