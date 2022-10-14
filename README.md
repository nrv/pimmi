# PIMMI : Python IMage MIning
Library allowing visual search in a corpus of images, from Twitter... or elsewhere.

The basic features are :
- SIFT interest points ([OpenCV](https://opencv.org/) implementation)
- efficient similarity search ([FAISS](https://github.com/facebookresearch/faiss) implementation)
- clustering via graph community detection (#TODO) 
- multithreaded

Very preliminary stuff for now.

## Install with pyenv and pip
```bash
pyenv virtualenv 3.7.0 pimmi-env
pyenv activate pimmi-env
pip install -U pip
pip install pimmi
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
```

Happy hacking !



