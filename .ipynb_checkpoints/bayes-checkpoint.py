from bayes_weight import load_bayes_weight, get_weight, load_pre_weight
import numpy as np


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def get_label(obj):
    max_conf = 0
    label_index = 0
    for i in range(5, len(obj)):
        conf = obj[i]
        if max_conf < conf:
            max_conf = conf
            label_index = i - 5
    return label_index


def get_strong_objects(image_pred, strong_thres):
    image_pred_strong = image_pred[image_pred[:, 4] * image_pred[:, 5:].max(1)[0] >= strong_thres]
    classes = load_classes("data/coco.names")

    sure_objects = []
    for obj in image_pred_strong:
        label_index = get_label(obj)
        sure_objects.append(label_index)

    return sure_objects


def get_possible_labels(obj, sure_objects):
    argue = False
    max_label = get_label(obj)
    possible_labels = [[max_label, obj[max_label+5]]]
    for i in range(80):
        if obj[max_label+5] - obj[i + 5] <= 0.6 * obj[max_label+5] and i != max_label:
            argue = True
            possible_labels.append([i, obj[i + 5]])

    return argue, possible_labels


def re_calculate_objects(image_pred,  weak_thres, sure_objects):
    pre_weight = load_pre_weight()
    weight = load_bayes_weight("coco")
    count = 0
    # half mAP: 0.5601068463908836(bayes) mAP: 0.5552795464218406(origin)
    # full mAP: 0.514102375905084(bayes) 0.5145143465508358(origin)
    for obj in image_pred:
        if obj[4] > weak_thres and get_label(obj) not in sure_objects:
            argue, possible_labels = get_possible_labels(obj,sure_objects)
            if argue:
                p_bayes = []
                bayes_sum = 0
                best_pro = 0
                best_label = 0
                for label in possible_labels:
#                     p_y = pre_weight[label[0]]
                    p_y = label[1]
                    p_xy = 1
                    for sure_obj in set(sure_objects):
                        p_xy *= weight[label[0]][sure_obj]
                    p_bayes.append([label[0], p_y * p_xy])
                for bayes_item in p_bayes:
                    bayes_sum += bayes_item[1]
                    if bayes_item[1] > best_pro:
                        best_pro = bayes_item[1]
                        best_label = bayes_item[0]
                bayes_output = [best_label, best_pro/bayes_sum+1e-16]
                obj[best_label+5] = bayes_output[1]
        image_pred[count] = obj
        count += 1

    return image_pred


# deal with single image with multiple objects
def bayes_factorize(image_pred, strong_thres, weak_thres):
    sure_objects = get_strong_objects(image_pred, strong_thres)
    output = re_calculate_objects(image_pred, weak_thres, sure_objects)

    return output
