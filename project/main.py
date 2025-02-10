import os

from tensorflow.python.ops.gen_experimental_dataset_ops import load_dataset
from graph_construction.create_graph_by_tf_idf import construct_graph
from graph_construction.preprocess import filter_data, write_to_csv_by_label, write_cleaned_data

os.OMP_NUM_THREADS = 1

os.environ["OMP_NUM_THREADS"] = "1"


# preprocess_1
ds = load_dataset('arrmlet/reddit_dataset_123456')
train_ds = ds['train']
filtered_ds = train_ds.filter(filter_data)
column_keep = ['text', 'label', 'username_encoded']
selected_ds = filtered_ds.remove_columns([col for col in filtered_ds.column_names if col not in column_keep])
write_to_csv_by_label(selected_ds, 'raw_data')

# preprocess_2
source_folder2 = r'dataset/raw_data'
destination_folder2 = r'dataset/cleaned_data'
write_cleaned_data(source_folder2, destination_folder2)

# graph_edges_construction
source_folder = r'dataset/cleaned_data'
construct_graph(source_folder, 'dataset/edges', th = 0)
construct_graph(source_folder, 'dataset/edges_4', th = 0.4)

