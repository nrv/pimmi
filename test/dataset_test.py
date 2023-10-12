import requests
import os


class TestDataset(object):
    def dataset_test(self):

        try:
            url_folder = ["https://github.com/nrv/pimmi/tree/main/demo_dataset/dataset1",
                          "https://github.com/nrv/pimmi/tree/main/demo_dataset/small_dataset/0", "https://github.com/nrv/pimmi/tree/main/demo_dataset/small_dataset/1"]

            urls = ["https://raw.githubusercontent.com/nrv/pimmi/main/demo_dataset/dataset1", "https://raw.githubusercontent.com/nrv/pimmi/main/demo_dataset/small_dataset/0",
                    "https://raw.githubusercontent.com/nrv/pimmi/main/demo_dataset/small_dataset/1"]
            names = ['dataset1', 'small_dataset/0', 'small_dataset/1']
            nb_images = [1000, 10, 1]
            for index, url in enumerate(urls):
                numero_image = 0
                r = requests.get(url_folder[index])
                assert r.status_code != 404, 'The folder dataset ' + \
                    names[index]+' is missing'

                url_base = os.path.join(url, "000")
                count = 0
                if index == 2:
                    numero_image = 10
                while count < nb_images[index]:
                    if 0 <= numero_image <= 9:
                        url = url_base+"00"+str(numero_image)+".jpg"
                    if 10 <= numero_image <= 99:
                        url = url_base+"0"+str(numero_image)+".jpg"
                    if 100 <= numero_image:
                        url = url_base+str(numero_image)+".jpg"
                    contents = requests.get(url)
                    assert contents.status_code != 404, 'Some images are missing in the dataset ' + \
                        names[index]
                    count += 1
                    numero_image += 1
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)
