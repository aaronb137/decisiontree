import numpy as np
import decision_trees as dt

file_name = "cat_dog_data.csv"
data = np.genfromtxt(file_name, dtype=str, delimiter=',')

# array builder
def build_nparray(data):
    feature_vectors, label_points = [], []
    values = data[1:][0:]

    for row in values:
        label_points.append(row[-1])
        feature_vectors.append(row[:-1])

    feature_vectors = np.array(feature_vectors, dtype=np.float32)
    label_points = np.array(label_points, dtype=np.int32)
    
    return feature_vectors, label_points

# list builder
def build_list(data):
    feature_vectors, label_points = [], []
    values = data[1:][0:]

    for row in values:
        label_points.append(row[-1])
        feature_vectors.append(row[:-1])
    
    feature_vectors = np.array(feature_vectors, dtype=np.float32)
    label_points = np.array(label_points, dtype=np.int32)

    feature_vectors = feature_vectors.tolist()
    label_points = label_points.tolist()

    return feature_vectors, label_points

# dictionary builder
def build_dict(data):
    feature_dict = {}
    label_dict = {}

    header_values = data[0, :-1]
    feature_values = data[1:, :-1]
    output_values = data[1:, -1]

    feature_values = np.array(feature_values, dtype=np.float32)
    output_values = np.array(output_values, dtype=np.int32)
    feature_values = feature_values.tolist()
    output_values = output_values.tolist()

    for array in feature_values:
        array_dict = {}
        for value_index, value in enumerate(array):
            array_dict[header_values[value_index]] = value
        feature_dict[len(feature_dict)] = array_dict
    
    for i in range(len(output_values)):
        label_dict[len(label_dict)] = output_values[i]

    return feature_dict, label_dict