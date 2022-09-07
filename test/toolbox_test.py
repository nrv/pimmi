from os.path import join, dirname
from pimmi import create_index_mt

SMALL_DATASET_DIR = join(dirname(dirname(__file__)), 'demo_dataset', 'small_dataset')


class TestIndex(object):
    def test_create_index(self):
        pass
