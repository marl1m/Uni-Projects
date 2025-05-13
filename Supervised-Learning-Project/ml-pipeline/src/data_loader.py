
import pandas as pd

def load_data(train_path, test_path, submission_path):
    return (
        pd.read_csv(train_path),
        pd.read_csv(test_path),
        pd.read_csv(submission_path),
    )
