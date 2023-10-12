import os
from github import Github, Repository, ContentFile
import requests


def download_demo(dataset, dir):
    numero_image = 0
    if dataset == 'dataset1':
        urls = [
            "https://raw.githubusercontent.com/nrv/pimmi/main/demo_dataset/dataset1/000"]
        nb_images = [1000]
    if dataset == 'small_dataset':
        urls = ["https://raw.githubusercontent.com/nrv/pimmi/main/demo_dataset/small_dataset/0/000",
                "https://raw.githubusercontent.com/nrv/pimmi/main/demo_dataset/small_dataset/1/000"]
        nb_images = [10, 1]

        if not os.path.isdir(os.path.join(dir, "0")):
            os.makedirs(os.path.join(dir, "0"))
        if not os.path.isdir(os.path.join(dir, "1")):
            os.makedirs(os.path.join(dir, "1"))

        dirs = [os.path.join(dir, "0"), os.path.join(dir, "1")]
    try:
        for index, url_base in enumerate(urls):
            count = 0
            while count < nb_images[index]:
                if 0 <= numero_image <= 9:
                    url = url_base+"00"+str(numero_image)+".jpg"
                    name = "00000"+str(numero_image)+".jpg"
                if 10 <= numero_image <= 99:
                    url = url_base+"0"+str(numero_image)+".jpg"
                    name = "0000"+str(numero_image)+".jpg"
                if 100 <= numero_image:
                    url = url_base+str(numero_image)+".jpg"
                    name = "000"+str(numero_image)+".jpg"
                contents = requests.get(url)
                if dataset == 'small_dataset':
                    if index == 0:
                        dir = dirs[0]
                    if index == 1:
                        dir = dirs[1]
                open(os.path.join(dir, name), "wb").write(contents.content)
                count += 1
                numero_image += 1
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
