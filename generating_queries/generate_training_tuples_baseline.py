import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle
import random

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "/home/kevin/DATA/"

runs_folder = "oxford/"
filename = "pointcloud_locations_20m_10overlap.csv"
pointcloud_fols = "/output_files_2k/"

all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
print(len(all_folders))
folders = []

# All runs are used for training (both full and partial)
index_list = range(len(all_folders))
print("Number of runs: " + str(len(index_list)))
# # ['2014-05-19-13-20-57', ..., '2015-11-12-11-22-05']
for index in index_list:
    if not all_folders[index].startswith('.'):
        folders.append(all_folders[index])

# #####For training and test data split#####
x_width = 150
y_width = 150
p1 = [5735712.768124, 620084.402381]
p2 = [5735611.299219, 620540.270327]
p3 = [5735237.358209, 620543.094379]
p4 = [5734749.303802, 619932.693364]
p = [p1, p2, p3, p4]


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if northing + x_width > point[0] > northing - x_width \
                and easting - y_width < point[1] < easting + y_width:
            in_test_set = True
            break
    return in_test_set

def construct_query_dict(df_centroids, filename):
    tree = KDTree(df_centroids[['northing', 'easting']])
    # index
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=10)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=50)
    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        # np.setdiff1d: Return the unique values in ar1 that are not in ar2.
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        negatives = np.setdiff1d(df_centroids.index.values.tolist(), ind_r[i]).tolist()
        random.shuffle(negatives)
        queries[i] = {"query": query, "positives": positives, "negatives": negatives}
    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

for folder in folders:

    df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')

    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    for index, row in df_locations.iterrows():
        row['file'] = runs_folder + folder + pointcloud_fols + "cloud_" + str(index) + "_ndt.bin"
        if check_in_test_set(row['northing'], row['easting'], p, x_width, y_width):
            df_test = df_test.append(row, ignore_index=True)
        else:
            df_train = df_train.append(row, ignore_index=True)

print("Number of training submaps: " + str(len(df_train['file'])))
print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
construct_query_dict(df_train, "training_queries_baseline.pickle")
construct_query_dict(df_test, "test_queries_baseline.pickle")

