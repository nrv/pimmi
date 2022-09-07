# =============================================================================
# Pimmi Toolbox Unit Tests
# =============================================================================
from os.path import join, dirname
from pimmi.toolbox import get_all_images

SMALL_DATASET_DIR = join(dirname(dirname(__file__)), 'demo_dataset', 'small_dataset')
NB_IMAGES = 11


class TestToolbox(object):
    def test_get_all_images(self):
        assert len(get_all_images(SMALL_DATASET_DIR)) == NB_IMAGES
