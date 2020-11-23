import os
import sys
import json
import random
import argparse


def generateSets(datasetFolder, dataFormat, trainingSetRatio):
    os.makedirs('data', exist_ok=True)
    subFolder = [os.path.join(datasetFolder, sub) for sub in os.listdir(
        datasetFolder) if os.path.isdir(os.path.join(datasetFolder, sub))]
    trainingSet = dict()
    testSet = dict()
    for folder in subFolder:
        tmp = [dataFile for dataFile in os.listdir(
            folder) if (dataFile.endswith(dataFormat) and dataFile[-5] != 't' and int(dataFile[13:16]) < 90)]
        random.shuffle(tmp)
        num_in_trainingset = int(len(tmp)*trainingSetRatio)
        trainingSet[folder] = tmp[0:num_in_trainingset]
        testSet[folder] = tmp[num_in_trainingset+1:]
    with open(os.path.join('data', 'training_set.json'), 'w') as train:
        train.write(json.dumps(trainingSet, indent=4, sort_keys=True))
    with open(os.path.join('data', 'test_set.json'), 'w') as test:
        test.write(json.dumps(testSet, indent=4, sort_keys=True))
    if os.path.exists('data/data.npz'):
        os.remove('data/data.npz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datasetFolder", type=str,
                        default='E:\dataset\CroppedYale')
    parser.add_argument("-f", "--format", type=str,
                        default="pgm", help="data format")
    parser.add_argument("-r", "--ratio", type=float, default=0.8,
                        help="the ratio of trainingSet, default 0.8")
    args = parser.parse_args()

    generateSets(args.datasetFolder, args.format, args.ratio)
