import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from check_structure import check_existing_file
import os

def main():
    """ Runs data normalising script to normalise X_train and X_test dataset (saved in../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('normalizing X_train and X_test data')

    input_filepath_X_train = "./data/processed_data/X_train.csv"
    input_filepath_X_test = "./data/processed_data/X_test.csv"
    output_filepath = "./data/processed_data/"

    normalize_data(input_filepath_X_train, input_filepath_X_test, output_filepath)

def normalize_data(input_filepath_X_train, input_filepath_X_test, output_filepath):
    # Import dataset
    X_train = import_dataset(input_filepath_X_train, sep=",")
    X_test = import_dataset(input_filepath_X_test, sep=",")

    # Normalise
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train_scaled, X_test_scaled, output_filepath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            pd.DataFrame(file).to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()