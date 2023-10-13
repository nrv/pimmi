import os


class TestDataset(object):
    def test_dataset(self):
        folders = ["demo_dataset/dataset1",
                   "demo_dataset/small_dataset/0", "demo_dataset/small_dataset/1"]

        names = ['dataset1', 'small_dataset/0', 'small_dataset/1']
        nb_images = [1598, 10, 1]
        for index, folder in enumerate(folders):
            numero_image = 0
            assert os.path.exists(folder), 'The folder dataset ' + \
                names[index]+' is missing'

            count = 0
            if index == 2:
                numero_image = 10
            while count < nb_images[index]:

                file = os.path.join(folder, str(
                    str(numero_image).zfill(6)) + ".jpg")

                assert os.path.exists(file), 'Some images are missing in the dataset ' + \
                    names[index]
                count += 1
                numero_image += 1
