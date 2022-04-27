# PIMMI : Python IMage MIning
Library allowing visual search in a corpus of images, from Twitter... or elsewhere.

SIFT interest points, clustering, based on OpenCV and Faiss, multithreaded.

Very preliminary stuff for now.


## Demo
```bash
# Install dependencies
pip install -r requirements.txt

# --- Play with a very small dataset
# Create a default index structure and fill it with the demo dataset  
python3 pimmi/index_dataset.py --action fill --thread 16 --index "IVF1024,Flat" --save_faiss index/small_dataset.ivf1024 --images_dir demo_dataset/small_dataset

# Query the same dataset on this index
python3 pimmi/query_dataset.py --simple --thread 16 --load_faiss index/small_dataset.ivf1024 --save_mining index/small_dataset.ivf1024.mining --images_mining --images_root demo_dataset/dataset1


# --- Play with the demo dataset 1
python3 pimmi/index_dataset.py --action fill --thread 16 --index "IVF1024,Flat" --save_faiss index/dataset1.ivf1024 --images_dir demo_dataset/dataset1
python3 pimmi/query_dataset.py --thread 16 --load_faiss index/dataset1.ivf1024 --save_mining index/dataset1.ivf1024.mining --images_mining --images_root demo_dataset/dataset1

# Post process the mining results in order to visualize them
python3 pimmi/fuse_query_results.py
python3 pimmi/generate_cluster_viz.py
```

Happy hacking !



