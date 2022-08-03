import pytest
import pimmi.pimmi_utils as ut

class TestGrid(object):
    def test_grid_1(self):
        image_id = 176584
        width = 640
        height = 480
        x = 152
        y = 349
        point_id = ut.point_to_full_id(image_id, [x, y], width, height)
        recovered_image_id = ut.full_id_to_image_id(point_id)
        assert image_id == recovered_image_id
        recovered_coord = ut.full_id_to_coord(point_id)
        assert x == recovered_coord[0, 0]
        assert y == recovered_coord[0, 1]
        grid_id = ut.full_id_to_grid_id(point_id)
        recovered_coord = ut.grid_id_to_coord(grid_id)
        assert x == recovered_coord[0, 0]
        assert y == recovered_coord[0, 1]