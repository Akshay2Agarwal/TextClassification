import pandas as pd
from deep_learn import constants


def fetch_imdb_train_data() -> pd.DataFrame:
    return pd.read_csv(constants.IMDB_TRAINING_FILE, sep='\t')


def fetch_imdb_test_data() -> pd.DataFrame:
    return pd.read_csv(constants.IMDB_TEST_FILE, sep='\t')