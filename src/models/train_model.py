import pandas as pd
import numpy as np
import joblib
from pathlib import Path

print(f"joblib version={joblib.__version__}")

def main(repo_path):
    # Load features and target data
    X_train_scaled = pd.read_csv("./data/processed_data/X_train_scaled.csv",sep=',')
    y_train = pd.read_csv("./data/processed_data/y_train.csv",sep=',')
    y_train = np.ravel(y_train)

    # load best alpha value
    best_ridge = joblib.load(repo_path / "models/ridge_best_params.pkl")

    # fit the grid search
    best_ridge.fit(X_train_scaled,y_train)

    # Save the trained model to a file
    model_filename = repo_path / "models/ridge_trained.pkl"
    joblib.dump(best_ridge, model_filename)
    
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)
