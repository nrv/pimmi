# =============================================================================
# Pimmi SIFT Points Unit Tests
# =============================================================================
import numpy as np

from pimmi.pimmi import extract_sift
from pimmi.toolbox import Sift
from test.utils import load_sifts_from_file, EXAMPLE_IMAGE_FILE, IMAGE_PATH, kp_fieldnames, prm

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 341


class TestSift(object):
    def test_extract_sift(self):
        sift = Sift(prm.sift_nfeatures, prm.sift_nOctaveLayers, prm.sift_contrastThreshold, prm.sift_edgeThreshold,
                    prm.sift_sigma, prm.nb_threads)

        assert extract_sift("wrong_path.jpg", 0, sift) == (None, None, None, None, None)

        ids, desc, kp = load_sifts_from_file(EXAMPLE_IMAGE_FILE)

        tested_ids, tested_kp, tested_desc, tested_width, tested_height = extract_sift(IMAGE_PATH, 0, sift)

        assert np.array_equal(desc, tested_desc)
        assert np.array_equal(ids, tested_ids)
        assert IMAGE_HEIGHT == tested_height
        assert IMAGE_WIDTH == tested_width
        for kp_item, tested_kp_item in zip(kp, tested_kp):
            assert kp_item.pt[0] == tested_kp_item.pt[0]
            assert kp_item.pt[1] == tested_kp_item.pt[1]
            for attr in kp_fieldnames[2:]:
                if attr == "angle":
                    assert round(float(getattr(kp_item, attr)), 3) == round(float(getattr(tested_kp_item, attr)), 3)
                else:
                    assert getattr(kp_item, attr) == getattr(tested_kp_item, attr)
