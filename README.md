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

### Initial authors
- [Béatrice Mazoyer](https://bmaz.github.io/)
- [Nicolas Hervé](http://herve.name)


## Install with pyenv and pip
```bash
pyenv virtualenv 3.7.0 pimmi-env
pyenv activate pimmi-env
pip install -U pip
pip install pimmi
```

## Install from source
```bash
python3 -m venv /tmp/pimmi-env
source /tmp/pimmi-env/bin/activate
pip install -U pip
git clone git@github.com:nrv/pimmi.git
cd pimmi
pip install -r requirements.txt 
pip install -e .
```

## Demo
```bash
# --- Play with the demo dataset 1
# Create a default index structure and fill it with the demo dataset. An 'index' directory will be created,
# it will contain the 2 files of the pimmi index : dataset1.IVF1024,Flat.faiss and 
# dataset1.IVF1024,Flat.meta
pimmi fill demo_dataset/dataset1 dataset1

# Query the same dataset on this index, the results will be stored in 
# index/dataset1.IVF1024,Flat.mining_000000.csv
pimmi query demo_dataset/dataset1 dataset1

# Post process the mining results in order to visualize them
pimmi clusters dataset1

# You can also play with the configuration parameters. First, generate a default configuration file
pimmi create-config my_pimmi_conf.yml

# Then simply use this configuration file to relaunch the mining steps (erasing without prompt the 
# previous data)
pimmi fill --erase --force --config-path my_pimmi_conf.yml demo_dataset/dataset1 dataset1
pimmi query --config-path my_pimmi_conf.yml demo_dataset/dataset1 dataset1
pimmi clusters --config-path my_pimmi_conf.yml dataset1
```

## Test on the Copydays dataset
You can find the dataset explanations [here](https://lear.inrialpes.fr/~jegou/data.php#copydays). Unfortunately, the data files are not available anymore, you can get them from [web archive](http://web.archive.org/web/20181015092553if_/http://pascal.inrialpes.fr/data/holidays/).

Create a project structure and uncompress all the files in the same images directory.

```
copydays
└───index
└───images
    └───crop
    │   └───crops
    │       └───10
    │       └───15
    │       └───20
    │       └───30
    │       └───40
    │       └───50
    │       └───60
    │       └───70
    │       └───80
    └───original
    └───jpeg
    │   └───jpegqual
    │       └───3
    │       └───5
    │       └───8
    │       └───10
    │       └───15
    │       └───20
    │       └───30
    │       └───50
    │       └───75
    └───strong
```

You can then play with the different parameters and evaluate the results. If you want to loop over several parameters to optimize your settings, you may have a look at eval_copydays.sh. 
```bash
cd scripts
mkdir index
pimmi --sift-nfeatures 1000 --index-type IVF1024,Flat fill /path/to/copydays/images/ copydays
pimmi --query-sift-knn 1000 --query-dist-ratio-threshold 0.8 --index-type IVF1024,Flat query /path/to/copydays/images/ copydays 
pimmi --index-type IVF1024,Flat --algo components clusters copydays
python copydays_groundtruth.py /path/to/copydays/images/ index/copydays.IVF1024,Flat.mining.clusters.csv
pimmi eval index/copydays.IVF1024,Flat.mining.groundtruth.csv --query-column image_status
```
```
cluster precision: 0.9924645454677958
cluster recall: 0.7406974660159374
cluster f1: 0.7856752626502786
query average precision: 0.8459113427266295
```
Happy hacking !



