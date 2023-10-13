# PIMMI : Python IMage MIning

PIMMI is a software that performs visual mining in a corpus of images. Its main objective is to find all copies,
total or partial, in large volumes of images and to group them together. Our initial goal is to study the reuse
of images on social networks (typically, our first use is the propagation of memes on Twitter). However, we believe
that its use can be much wider and that it can be easily adapted for other studies. The main features of PIMMI
are therefore :

- ability to process large image corpora, up to several millions files
- be robust to some modifications of the images, typical of their reuse on social networks (crop, zoom,
  composition, addition of text, ...)
- be flexible enough to adapt to different use cases (mainly the nature and volume of the image corpora)

PIMMI is currently only focused on visual mining and therefore does not manage metadata related to images.
The latter are specific to each study and are therefore outside our scope. Thus, a study using PIMMI
will generally be broken down into several steps:

1. constitution of a corpus of images (jpg and/or png files) and their metadata
2. choice of PIMMI parameters according to the criteria of the corpus
3. indexing the images with PIMMI and obtaining clusters of reused images
4. exploitation of the clusters by combining them with the descriptive metadata of the images

PIMMI relies on existing technologies and integrates them into a simple data pipeline:

1. Use well-established local image descriptors (Scale Invariant Feature Transform: SIFT) to represent images
   as sets of keypoints. Geometric consistency verification is also used. ([OpenCV](https://opencv.org/) implementation
   for both).
2. To adapt to large volumes of images, it relies on a well known vectors indexing library that provides some
   of the most efficient algorithms implementations ([FAISS](https://github.com/facebookresearch/faiss)) to query
   the database of keypoints.
3. Similar images are grouped together using standard community detection algorithms on the graph of similarities.

PIMMI is a library developed in Python, which can be used through a command line interface. It is multithreaded.
A rudimentary web interface to visualize the results is also provided, but more as an example than for
intensive use ([Pimmi-ui](https://github.com/nrv/pimmi-ui)).

The development of this software is still in progress : we warmly welcome beta-testers, feedback,
proposals for new features and even pull requests !

## Authors

- [Béatrice Mazoyer](https://bmaz.github.io/)
- [Nicolas Hervé](http://herve.name)

## Installation

Pimmi requires Python 3 to be installed. If Python 3 is not installed on your computer, we recommend installing the distribution provided by Miniconda: https://docs.conda.io/projects/miniconda/en/latest/#quick-command-line-install

We recommend installing Pimmi in a virtual environment. The installation scenarios below provide instructions for installing Pimmi with [conda](#install-with-conda) (if you have Miniconda or Anaconda installed), with [venv](#install-with-venv) or with [pyenv-virtualenv](#install-with-pyenv-virtualenv-and-pip). If you are using another virtual environment management system, simply create a new environment, activate it and run:

```bash
pip install pimmi
```

### Install with conda

```bash
conda create --name pimmi-env
conda activate pimmi-env
pip install -U pip
pip install pimmi
```

### Install with venv

```bash
python3 -m venv /tmp/pimmi-env
source /tmp/pimmi-env/bin/activate
pip install -U pip
pip install pimmi

```

### Install with pyenv-virtualenv

```bash
pyenv virtualenv 3.8.0 pimmi-env
pyenv activate pimmi-env
pip install -U pip
pip install pimmi
```

## Demo

```bash
# --- Play with the demo dataset 1
# Download the demo dataset, it will be loaded in the folder demo_dataset
# You can choose between small_dataset and dataset1.
# small_dataset contains 10 images and dataset contains 1000 images, it takes 2 minutes to be downloaded.

pimmi download_demo dataset1

# Create a default index structure and fill it with the demo dataset. A directory named my_index will be created,
# it will contain the 2 files of the pimmi index : index.faiss and index.meta
pimmi fill demo_dataset/dataset1 my_index

# Query the same dataset on this index, the results will be stored in
# result_query.csv
pimmi query demo_dataset/dataset1 my_index -o result_query.csv

# Post process the mining results in order to visualize them
pimmi clusters my_index result_query.csv

# You can also play with the configuration parameters. First, generate a default configuration file
pimmi create-config my_pimmi_conf.yml

# Then simply use this configuration file to relaunch the mining steps (erasing without prompt the
# previous data)
pimmi fill --erase --force --config-path my_pimmi_conf.yml demo_dataset/dataset1 dataset1
pimmi query --config-path my_pimmi_conf.yml demo_dataset/dataset1 dataset1
pimmi clusters --config-path my_pimmi_conf.yml dataset1
```

## Test on the Copydays dataset

Unfortunately, the data files and the dataset explanations are not available anymore, you can get them from web archive with this [link](http://web.archive.org/web/20181015092553if_/http://pascal.inrialpes.fr/data/holidays/) for the data files and with this [link]https://web.archive.org/web/20170325224315/https://lear.inrialpes.fr/people/jegou/data.php for dataset explanations.

### Download the dataset

Download the 4 following gunzip folders : copydays_crop.tar.gz, copydays_jpeg.tar.gz, copydays_original.tar.gz, copydays_strong.tar.gz.
Create a project structure and uncompress all the files downloaded in the same images directory.

```
copydays
└───images
    └───copydays_crop
    └───original
    └───jpegqual
    └───copydays_strong
```

### Clone the repository

You can then play with the different parameters and evaluate the results. If you want to loop over several parameters to optimize your settings, you may have a look at eval_copydays.sh.
To be able to do so, you should have access to the "script" folder, which is in the github repo. Therefore, if you want to reproduce the results, you should clone the repository .

### Commands to reproduce the results

```bash
cd copydays
pimmi --sift-nfeatures 1000 --index-type IVF1024,Flat fill images/ my_index_folder
pimmi --query-sift-knn 1000 --query-dist-ratio-threshold 0.8 --index-type IVF1024,Flat query images my_index_folder -o result_query.csv
pimmi --index-type IVF1024,Flat --algo components clusters my_index_folder result_query.csv -o clusters.csv
python ../scripts/copydays_groundtruth.py images/ clusters.csv
pimmi eval groundtruth.csv --query-column image_status
```

### Results :

```
cluster precision: 0.98650288140734
cluster recall: 0.7441110375823754
cluster f1: 0.7838840961245362
query average precision: 0.839968152866242
```

## Troubleshooting

### Error while installing faiss-cpu for macOS > 12

```
error: command '/usr/local/bin/swig' failed with exit code 1
```

The installation of pimmi requires the package faiss-cpu. However, on macOS > 12 this package cannot be installed by pip. (https://github.com/facebookresearch/faiss/issues/2868)
To fix this issue, please follow these steps:

Install Miniconda :
https://docs.conda.io/projects/miniconda/en/latest/#quick-command-line-install

Create and activate a virtual environnement:

```
conda create --name testenv1
conda activate testenv1
```

In this virtual environment, install faiss-cpu:

```
conda install -c pytorch faiss-cpu
```

And then you should be able to install pimmi:

```
pip install pimmi
```

### I have another error

Please submit an issue [here](https://github.com/nrv/pimmi/issues)

## Contribute

Pull requests are welcome! Please find below the instructions to install a development version.

### Install from source

```bash
python3 -m venv /tmp/pimmi-env
source /tmp/pimmi-env/bin/activate
pip install -U pip
git clone git@github.com:nrv/pimmi.git
cd pimmi
pip install -r requirements.txt
pip install -e .
```

### Linting and tests

To lint the code and run the unit tests you can use the following commands:

```bash
# Only linter
make lint

# Only unit tests
make test

# Both
make
```
