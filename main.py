import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


if __name__=="__main__":
    random = 901
    cells = load_breast_cancer()
    X = cells.data
    y = cells.target

    # View data as dataframe
    cells_df = load_breast_cancer(as_frame=True).frame

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random, stratify=y)
    
    # Classifier
    classifier = MLPClassifier()

    # Hyperperameter grid, using "kitchen sink" approach to find a high scoring combination of hyperparameters
    param_grid = {
        "random_state": [random],
        "activation":['identity', 'logistic', 'tanh', 'relu'],
        "solver":['lbfgs','sgd','adam'],
        "learning_rate":['constant','invscaling','adaptive'],
        "max_iter":[100000000],
        "early_stopping": [True]
    }

    # cv layers in search set at default of 5
    hyper_parameter_search = GridSearchCV(classifier, param_grid, refit=True)
    pipe = make_pipeline(StandardScaler(), hyper_parameter_search)
    
    pipe.fit(X_train, y_train)

    y_predictions = pipe.predict(X_test)

    report = classification_report(y_test, y_predictions, target_names=cells.target_names)
    print(report)
