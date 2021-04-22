import os
import numpy as np


def deal_single_file(file_name):
    file = open(file_name, "r")
    labels = []
    for line in file.readlines():
        label = line.split(" ")[0]
        labels.append(int(label))
    return labels


def load_data():
    count = 0
    total_data = np.zeros([82081, 80], int)
    for root, dirs, files in os.walk("D:/final_project/PyTorch-YOLOv3-master/data/coco/labels/train2014"):
        for file in files:
            row_data = np.zeros([80], int)
            labels = deal_single_file("D:/final_project/PyTorch-YOLOv3-master/data/coco/labels/train2014/" + file)
            for label in labels:
                row_data[label] = 1
            total_data[count] = row_data
            count += 1
            if count % 100 == 0:
                print(str(count) + "/82081")
                # break

    return total_data


def get_pro_by_con(data, info):
    count = 0
    pro_by_con = []
    row_list = []
    for line in data:
        if line[info] == 1:
            row_list.append(count)
        count += 1
    if count == 0:
        info_data = np.zeros(80)
    info_data = data[[i for i in row_list], :]

    return info_data.sum(axis=0) / (len(info_data) + 1e-16)


def get_probability(data):
    af_probability = np.zeros([80, 80])
    for i in range(80):
        pro_row = get_pro_by_con(data, i)
        af_probability[i] = pro_row
    return af_probability


def calculate_probality_weight(pre_p,af_p):

    output = np.zeros([80,80])
    for i in range(len(af_p)):
        output[i] = af_p[i]/(pre_p + 1e-16)

    return output


if __name__ == "__main__":
    data = load_data()
    pre_p = data.sum(axis=0) / len(data)
    # dandan = get_pro_by_con(data, 0)
    # af_probability = get_probability(data)
    # output = calculate_probality_weight(pre_p=pre_p, af_p=af_probability)
    file = open("pre_weight.txt", "w")
    for i in range(80):
        file.write("%f " % pre_p[i])
        # for j in range(80):
        #     file.write("%f " % output[i][j])
        # file.write("\n")


    # print(pre_p)
