import xmltodict
import json
import os
import random


def parse_xml(file_name):
    xml_file = open("C:/Users/94419/Desktop/Annotations/" + file_name)
    xml_str = xml_file.read()
    xml_parse = xmltodict.parse(xml_str)
    json_str = json.dumps(xml_parse, indent=1)
    dict_ouput = json.loads(json_str)
    xml_file.close()

    return dict_ouput


def get_class():
    class_file = open("classes.names")
    labels = []
    count = 0
    for line in class_file.readlines():
        labels.append(line[:-2])
    labels[19] = "tvmonitor"
    class_file.close()

    return labels


def deal_single_file(file_name, labels):
    file = open("labels/" + file_name[:-4]+".txt", "w")
    dict_string = parse_xml(file_name)
    xsize = int(dict_string["annotation"]["size"]["width"])
    ysize = int(dict_string["annotation"]["size"]["height"])
    if type(dict_string["annotation"]["object"]) is not list:
        obj = dict_string["annotation"]["object"]
        index = labels.index(obj["name"])
        xmin = int(obj["bndbox"]["xmin"])
        xmax = int(obj["bndbox"]["xmax"])
        ymin = int(obj["bndbox"]["ymin"])
        ymax = int(obj["bndbox"]["ymax"])
        width = xmax - xmin
        height = ymax - ymin
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        output_string = str(index) + " " + "%.6f" % (x_center / xsize) + " " + "%.6f" % (
                y_center / ysize) + " " + "%.6f" % (width / xsize) + " " + "%.6f" % (height / ysize) + "\n"
        file.write(output_string)
    else:
        for obj in dict_string["annotation"]["object"]:
            index = labels.index(obj["name"])
            xmin = int(obj["bndbox"]["xmin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymin = int(obj["bndbox"]["ymin"])
            ymax = int(obj["bndbox"]["ymax"])
            width = xmax - xmin
            height = ymax - ymin
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            output_string = str(index) + " " + "%.6f" % (x_center / xsize) + " " + "%.6f" % (
                    y_center / ysize) + " " + "%.6f" % (width / xsize) + " " + "%.6f" % (height / ysize) + "\n"
            file.write(output_string)
    file.close()


def deal_all_files():
    labels = get_class()
    print(labels)
    count = 0
    for root, dirs, files in os.walk("C:/Users/94419/Desktop/Annotations"):
        for file in files:
            deal_single_file(file, labels)
            count += 1
            print(count)

    print("done")


if __name__ == "__main__":
    train_file = open("train2.txt", "w")
    valid_file = open("valid2.txt", "w")

    for root, dirs, files in os.walk("images"):
        for file in files:
            n = random.random()
            if n < 0.7:
                train_file.write("/home/featurize/data/voc/images/" + file + "\n")
                # train_file.write("D:/final_project/PyTorch-YOLOv3-master/data/custom/images/" + file + "\n")
            else:
                valid_file.write("/home/featurize/data/voc/images/" + file + "\n")
                # valid_file.write("D:/final_project/PyTorch-YOLOv3-master/data/custom/images/" + file + "\n")

    print("done")


