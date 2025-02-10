# Networks and Dynamics in Reddit Political Discussion


## Project Structure
- `requirements.txt`: Pip environment file.
- `data/`: datasets. See `data/README.md` for details .
- `analysis/`: analysis methods.
- `main.py`: processes a Reddit dataset and constructs graphs based on the processed data. The script performs data preprocessing, filtering, and graph construction using TF-IDF.
- `grave_construction/`: Source codes.
  - `create_graph_by_tf_idf.py`: Create matrix combining TF-IDF and SentenceTransformer, you can change parameter -- alpha, in this file to get different blending matrix.
  - `preprocess.py`: Preprocess raw data then save into cleaned_data folder.
- `grave2Vec/`: Source codes.
  - `main_graph2vec.py`: Including Graph2Vec embedding algorithm, supervised and unsupervised learning algorithm for analysis.

## Get started
The required libraries  can be found in `requirements.txt`.

# How to run
Use the following command to install the dependencies:
```bash
pip install -r requirements.txt
```
Use the following command to preprocess the data and construct networks:
```bash
python main.py
# You can change threshold in this file to create different networks
```
Use the following command to get embeddings and clustering, classification results:
```bash
python graph2Vec/main_graph2vec.py
# You can change dataset folder path in this file to get different results from different dataset
# You also can use different clustering algorithms by changing variables in this file.
```
Use the following command to get metrics for evaluations of networks in different labels:
```bash
python analysis.py
# You can change dataset folder in this file to get different results from different networks.
```