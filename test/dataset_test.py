import requests
import os


class TestDataset(object):
    def dataset_test(self):
        folders = ["demo_dataset/dataset1",
                   "demo_dataset/small_dataset/0", "demo_dataset/small_dataset/1"]

        names = ['dataset1', 'small_dataset/0', 'small_dataset/1']
        nb_images = [1000, 10, 1]
        for index, folder in enumerate(folders):
            numero_image = 0
            assert os.path.exists(folder), 'The folder dataset ' + \
                names[index]+' is missing'

            file_base = os.path.join(folder, "000")
            count = 0
            if index == 2:
                numero_image = 10
            while count < nb_images[index]:
                if 0 <= numero_image <= 9:
                    file = file_base+"00"+str(numero_image)+".jpg"
                if 10 <= numero_image <= 99:
                    file = file_base+"0"+str(numero_image)+".jpg"
                if 100 <= numero_image:
                    file = file_base+str(numero_image)+".jpg"
                assert os.path.exists(file), 'Some images are missing in the dataset ' + \
                    names[index]
                count += 1
                numero_image += 1
