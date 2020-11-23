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


def loadData(enable_cache, quiet):
    start = timer()
    cache_file = 'data/data.npz'
    if enable_cache and os.path.exists(cache_file):
        data = np.load(cache_file)
        training_data = data['arr_0']
        training_data_name = data['arr_1']
        training_data_range = data['arr_2']
        test_data = data['arr_3']
        test_data_name = data['arr_4']
        test_data_range = data['arr_5']
    else:
        training_data, training_data_name, training_data_range = getDataFromJSON(
            'data/training_set.json')
        test_data, test_data_name, test_data_range = getDataFromJSON(
            'data/test_set.json')

        if enable_cache:
            np.savez(cache_file, training_data, training_data_name,
                     training_data_range, test_data, test_data_name, test_data_range)
    end = timer()
    load_time = end - start
    if not quiet:
        print(str(training_data.shape[0]), "training images and", str(
            test_data.shape[0]), "testing images loaded in %.6f seconds!" % load_time)

    return training_data, training_data_name, training_data_range, test_data, test_data_name, test_data_range, load_time


def training(dim, neighbors, enable_cache, show_group_accuracy, quite):
    performance = dict()
    performance['dimension'] = dim
    performance['neighbors'] = neighbors

    # load data
    (training_data, training_data_name, training_data_range, test_data, test_data_name, test_data_range, load_time) = loadData(
        enable_cache, quite)
    performance['load_time'] = load_time

    # training data
    training_start = timer()
    pca = PCA(n_components=dim).fit(training_data)
    compressed_training_data = pca.transform(training_data)
    training_end = timer()
    training_time = training_end - training_start
    if not quiet:
        print('training complete in %.6f seconds! dimensions:%d' %
              (training_time, dim))
    performance['training_time'] = training_time

    # training result infomation
    correct_count = 0
    group_num = len(test_data_name)
    group_correct_count = [0]*group_num

    # testing data
    testing_start = timer()
    compressed_test_data = pca.transform(test_data)
    for index_of_test_data, data1 in enumerate(compressed_test_data, start=0):
        # for each test data, compute distance between it and all training data
        distance = list()
        for index_of_training_data, data2 in enumerate(compressed_training_data, start=0):
            distance.append(
                (index_of_training_data, np.linalg.norm(data1-data2)))
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
            group_correct_count[getGroupIndex(
                index_of_test_data, test_data_range)] += 1
    testing_end = timer()
    testing_time = testing_end - testing_start
    if not quiet:
        print("testing complete in %.6f seconds! neighbors:%d" %
              (testing_time, neighbors))
    performance['testing_time'] = testing_time

    # calculate accuracy
    accuracy = correct_count/test_data.shape[0]
    group_accuracy = list()
    for index in range(len(test_data_name)):
        group_accuracy.append(group_correct_count[index] /
                              (test_data_range[index+1] - test_data_range[index]))
    if not quiet:
        print('Accuracy: %.6f' % accuracy)
    performance['accuracy'] = accuracy
    performance['group_accuracy'] = group_accuracy
    if show_group_accuracy:
        showGroupAccuracy(test_data_name, group_accuracy)

    if quiet:
        print("dimension: %d, neighbors: %d, accuracy: %.6f" %
              (dim, neighbors, accuracy))
    return performance


if __name__ == '__main__':

    enable_cache = True
    show_group_accuracy = False
    quiet = True

    dimension_start = 1
    dimension_end = 1000
    dimension_gap = 5
    neighbors_start = 1
    neighbors_end = 2
    neighbors_gap = 1

    performance = list()
    loop_params = dict()
    best_acc = {"dimension": -1, "neighbors": -1, "accuracy": 0.0}
    loop_params['dimension'] = [x for x in range(dimension_start, dimension_end, dimension_gap)]
    loop_params['neighbors'] = [x for x in range(neighbors_start, neighbors_end, neighbors_gap)]

    for dimension in range(dimension_start, dimension_end, dimension_gap):
        for neighbor in range(neighbors_start, neighbors_end, neighbors_gap):
            res = training(dimension, neighbor, enable_cache,
                           show_group_accuracy, quiet)
            if res['accuracy'] > best_acc['accuracy']:
                best_acc['accuracy'] = res['accuracy']
                best_acc['dimension'] = res['dimension']
                best_acc['neighbors'] = res['neighbors']
            performance.append(res)

    print('best_acc:', best_acc)

    with open('data/performance.json', 'w') as performance_file:
        performance_file.write(json.dumps({"loop_params": loop_params, "performance": performance}, indent=4))
