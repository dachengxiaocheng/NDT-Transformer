import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random

#####For training and test data split#####
x_width = 150
y_width = 150

# For Oxford
p1 = [5735712.768124, 620084.402381]
p2 = [5735611.299219, 620540.270327]
p3 = [5735237.358209, 620543.094379]
p4 = [5734749.303802, 619932.693364]

p_dict = {"oxford": [p1, p2, p3, p4]}


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if (point[0] - x_width < northing and northing < point[0] + x_width and point[
            1] - y_width < easting and easting < point[1] + y_width):
            in_test_set = True
            break
    return in_test_set


##########################################

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name):
    database_trees = []
    test_trees = []
    count = 0
    count_2 = 0
    for folder in folders:
        df_database = pd.DataFrame(columns=['timestamp', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['timestamp', 'northing', 'easting'])

        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        for _, row in df_locations.iterrows():
            # entire business district is in the test set
            if output_name == "business":
                df_test = df_test.append(row, ignore_index=True)
            elif check_in_test_set(row['northing'], row['easting'], p, x_width, y_width):
                df_test = df_test.append(row, ignore_index=True)
                count_2 += 1
            df_database = df_database.append(row, ignore_index=True)
            count += 1
        database_tree = KDTree(df_database[['northing', 'easting']])
        test_tree = KDTree(df_test[['northing', 'easting']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)
    print(count)
    print(count_2)
    test_sets = []
    database_sets = []
    for folder in folders:
        print(folder)
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            row['file'] = runs_folder + folder + pointcloud_fols + "cloud_" + str(index) + "_ndt.bin"
            if output_name == "business":
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            elif check_in_test_set(row['northing'], row['easting'], p, x_width, y_width):
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                              'easting': row['easting']}
            
        database_sets.append(database)
        test_sets.append(test)
    print(len(database_sets))
    print(len(test_sets))
    
    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if (i == j):
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                
                index = tree.query_radius(coor, r=25)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()
    output_to_file(database_sets, output_name + '_evaluation_database.pickle')
    output_to_file(test_sets, output_name + '_evaluation_query.pickle')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "/home/kevin/DATA/"

folders = []
runs_folder = "oxford/"
path = os.path.join(base_path, runs_folder)
print(path)
all_folders = sorted(os.listdir(path))
index_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 31, 32, 33, 38, 43, 44]
for index in index_list:
    folders.append(all_folders[index - 1])

construct_query_and_database_sets(base_path, runs_folder, folders, "/output_files_2k_20/", "pointcloud_locations_20m.csv",
                                  p_dict["oxford"], "oxford")

