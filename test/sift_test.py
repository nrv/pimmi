# =============================================================================
# Pimmi SIFT Points Unit Tests
# =============================================================================
import numpy as np

from pimmi.pimmi import extract_sift
from pimmi.toolbox import Sift
from test.utils import load_sifts_from_file, EXAMPLE_IMAGE_FILE, IMAGE_PATH, kp_fieldnames, prm

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 341
ROUND_DECIMALS = 2


class TestSift(object):
    def test_extract_sift(self):
        sift = Sift(prm.sift_nfeatures, prm.sift_nOctaveLayers, prm.sift_contrastThreshold, prm.sift_edgeThreshold,
                    prm.sift_sigma, prm.nb_threads)

        assert extract_sift("wrong_path.jpg", 0, sift) == (None, None, None, None, None)

        ids, desc, kp = load_sifts_from_file(EXAMPLE_IMAGE_FILE)

        tested_ids, tested_kp, tested_desc, tested_width, tested_height = extract_sift(IMAGE_PATH, 0, sift)

        assert desc.shape == tested_desc.shape
        assert np.allclose(desc, tested_desc, atol=1.)
        assert np.array_equal(ids, tested_ids)
        assert IMAGE_HEIGHT == tested_height
        assert IMAGE_WIDTH == tested_width
        for kp_item, tested_kp_item in zip(kp, tested_kp):
            assert round(kp_item.pt[0], ROUND_DECIMALS) == round(tested_kp_item.pt[0], ROUND_DECIMALS)
            assert round(kp_item.pt[1], ROUND_DECIMALS) == round(tested_kp_item.pt[1], ROUND_DECIMALS)
            for attr in kp_fieldnames[2:]:
                assert round(float(getattr(kp_item, attr)), ROUND_DECIMALS) == round(float(getattr(tested_kp_item, attr)), ROUND_DECIMALS)
