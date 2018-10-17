import pandas as pd
from deep_learn import constants

class FetchData:

    def _fetch_imdb_train_data(self) -> pd.DataFrame:
        return pd.read_csv(constants.IMDB_TRAINING_FILE, sep='\t')

    def _fetch_imdb_test_data(self) -> pd.DataFrame:
        return pd.read_csv(constants.IMDB_TEST_FILE, sep='\t')