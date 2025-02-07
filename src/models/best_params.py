import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path

print(f"joblib version={joblib.__version__}")

def main(repo_path):
    # Load features and target data
    X_train_scaled = pd.read_csv("./data/processed_data/X_train_scaled.csv",sep=',')
    y_train = pd.read_csv("./data/processed_data/y_train.csv",sep=',')
    y_train = np.ravel(y_train)

    # define model (Linear Regression Ridge) 
    model = Ridge()

    # define parameters values
    parameters = {'alpha':[0.01,0.1,1,10,100]}

    # define the grid search
    ridge_gridcv = GridSearchCV(model,parameters, scoring='neg_mean_squared_error',cv=10)

    # fit the grid search
    ridge_gridcv.fit(X_train_scaled,y_train)

    # save best estimator
    best_params_file = repo_path / "models/ridge_best_params.pkl"
    joblib.dump(ridge_gridcv.best_estimator_,best_params_file,compress=True)
    print("Best parameters found and saved successfully.")

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)

