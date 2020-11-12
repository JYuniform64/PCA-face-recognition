import numpy as np
from sklearn.decomposition import PCA
import sys
import os
import json
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import pgm


def getDataFromJSON(json_file):
    with open(json_file, 'r') as f:
        raw_data = f.read()
    json_data = json.loads(raw_data)
    facesName = tuple(tmp.split('\\')[-1] for tmp in json_data)

    facesRange = list([0])
    for face in json_data:
        facesRange.append(facesRange[-1]+len(json_data[face]))

    data = list()
    for face in json_data:
        for image in json_data[face]:
            with open(os.path.join(face, image), 'rb') as pgmf:
                data += pgm.read_pgm_1d_array(pgmf)

    array = np.array(data)
    return array, facesName, facesRange


def sort_keys(tup):
    return tup[1]


def getGroupIndex(index, range):
    for index1, value in enumerate(range, start=0):
        if index < value:
            return index1-1


def getNameFromIndex(index, range, name):
    return name[getGroupIndex(index, range)]


def most_common(lst):
    return max(set(lst), key=lst.count)

def showGroupAccuracy(name, accuracy):
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    y_pos = np.arange(len(name))

    ax.barh(y_pos, accuracy, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(name)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('accuracy')
    ax.set_title('Group Accuracy')

    plt.show()

if __name__ == '__main__':
    # model arguments
    dim = 100
    neighbors =10

    # load data
    training_data, training_data_name, training_data_range = getDataFromJSON(
        'data\\training_set.json')
    test_data, test_data_name, test_data_range = getDataFromJSON(
        'data\\test_set.json')

    print('data loaded! all together ', str(training_data.shape[0]), " training images and ", str(
        test_data.shape[0]), " testing images.")

    # training data
    pca = PCA(n_components=dim).fit(training_data)
    compressed_training_data = pca.transform(training_data)

    print('training completed!')

    # training result infomation
    correct_count = 0
    group_num = len(test_data_name)
    group_correct_count = [0]*group_num

    # testing data
    compressed_test_data = pca.transform(test_data)
    for index_of_test_data, data1 in enumerate(compressed_test_data, start=0):
        # for each test data, compute distance between it and all training data
        distance = list()
        for index_of_training_data, data2 in enumerate(compressed_training_data, start=0):
            distance.append((index_of_training_data, np.linalg.norm(data1-data2)))
        distance = sorted(distance, key=sort_keys)

        # get nearest k neighbors
        names_of_nearest_neighbors = list()
        name_of_test_data = getNameFromIndex(
            index_of_test_data, test_data_range, test_data_name)
        
        for j in range(neighbors):
            names_of_nearest_neighbors.append(getNameFromIndex(
                distance[j][0], training_data_range, training_data_name))
        # print(name_of_test_data,names_of_nearest_neighbors, 'correct' if most_common(names_of_nearest_neighbors) == name_of_test_data else 'incorrect')

        # record recognition results
        if name_of_test_data == most_common(names_of_nearest_neighbors):
            correct_count += 1
            group_correct_count[getGroupIndex(index_of_test_data, test_data_range)] += 1

    # calculate accuracy
    group_accuracy = list()
    for index in range(len(test_data_name)):
        group_accuracy.append(group_correct_count[index] / \
            (test_data_range[index+1] - test_data_range[index]))
    print('Accuracy: ' + str(correct_count/test_data.shape[0]))
    showGroupAccuracy(test_data_name, group_accuracy)