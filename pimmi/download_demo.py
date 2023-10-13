import os
import requests
import logging
import sys
import time
from tqdm import tqdm


def download_demo(dataset, dir):
    logger = logging.getLogger("pimmi")
    numero_image = 0
    if dataset == 'dataset1':
        urls = [
            "https://raw.githubusercontent.com/nrv/pimmi/main/demo_dataset/dataset1/"]
        nb_images = [1598]
    elif dataset == 'small_dataset':
        urls = ["https://raw.githubusercontent.com/nrv/pimmi/main/demo_dataset/small_dataset/0/",
                "https://raw.githubusercontent.com/nrv/pimmi/main/demo_dataset/small_dataset/1/"]
        nb_images = [10, 1]

        if not os.path.isdir(os.path.join(dir, "0")):
            os.makedirs(os.path.join(dir, "0"))
        if not os.path.isdir(os.path.join(dir, "1")):
            os.makedirs(os.path.join(dir, "1"))

        dirs = [os.path.join(dir, "0"), os.path.join(dir, "1")]

    for index, url_base in enumerate(urls):
        count = 0
        for count in tqdm(range(nb_images[index])):
            name = str(str(numero_image).zfill(6)) + ".jpg"
            url = url_base+name
            contents = None
            for i in range(3):
                try:
                    contents = requests.get(url)
                    break
                except requests.exceptions.ConnectionError:
                    time.sleep(5)

            if not contents:
                logger.error(
                    "The connection failed. Please check your connection and try again.")
                sys.exit(1)

            if dataset == 'small_dataset':
                if index == 0:
                    dir = dirs[0]
                if index == 1:
                    dir = dirs[1]
            open(os.path.join(dir, name), "wb").write(contents.content)
            numero_image += 1
