import csv
import numpy as np
from utils import *
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_stat',
        default='./data/stat.csv',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--data_path',
        default='./data/test.csv',
        type=str,
        help='Directory path of results')
    parser.add_argument(
        '--result_path',
        default='/scratch/shared/beegfs/hchen/prediction/audioclassification/vggsound/resnet18/',
        type=str,
        help='metadata directory')
    return parser.parse_args() 

def main():
    args = get_arguments()
    classes = []
    data = []
    data2class = {}

    # load classes
    with open(args.data_stat) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            classes.append(row[0])
    classes = sorted(classes)

    # load test data
    with open(args.data_path) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            data2class[row[0].split('/')[-1]] = row[1]
            data.append(row[0])

    # placeholder for prediction and gt
    pred_array = np.zeros([len(data),len(classes)])
    gt_array = np.zeros([len(data),len(classes)])


    for count,item in enumerate(data):
        pred = np.load(args.result_path + item + '.npy')

        label = data2class[item.split('/')[-1]]
        label_index = []
        label_index.append(classes.index(label))
        
        pred_array[count,:] = pred
        gt_array[count,np.array(label_index)] = 1


    stats = calculate_stats(pred_array,gt_array)
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])

    print("mAP: {:.6f}".format(mAP))
    print("mAUC: {:.6f}".format(mAUC))
    print("dprime: {:.6f}".format(d_prime(mAUC)))

if __name__ == "__main__":
    main()

