import pandas as pd
from deeplearn import constants

class FetchData(object):

    def _fetch_imdb_train_data(self):
        return pd.read_csv(constants.IMDB_TRAINING_FILE, sep='\t')

    def _fetch_imdb_test_data(self):
        return pd.read_csv(constants.IMDB_TEST_FILE, sep='\t')