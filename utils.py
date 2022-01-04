from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
import os


data_root_path = 'datasets/'

METHOD_NAME = None
RECORD_FILE_NAME = None
DATA_DIM = 20
OUTLIER_RATE = 3 / 100
DATAID = -1
NUM_VIEWS = 2


def get_view_fea():
    avg_num = int(DATA_DIM / NUM_VIEWS)
    left_num = DATA_DIM % NUM_VIEWS
    fea_num = [avg_num for a in range(NUM_VIEWS)]
    for i in range(left_num):
        fea_num[NUM_VIEWS - 1 - i] += 1
    start = 0
    end = 0
    s = [0 for a in range(NUM_VIEWS)]
    e = [0 for a in range(NUM_VIEWS)]
    for v in range(NUM_VIEWS):
        end = start + fea_num[v]
        s[v] = start
        e[v] = end
        start = end
    return s, e

def set_num_views(n):
    global NUM_VIEWS
    NUM_VIEWS = n

def set_record_file(f):
    global RECORD_FILE_NAME
    RECORD_FILE_NAME = f


def set_data_id(i):
    global DATAID
    DATAID = i


def set_file_name(f):
    global FILE_NAME
    FILE_NAME = f


def set_method_name(n):
    global METHOD_NAME
    METHOD_NAME = n


def set_dim(d):
    global DATA_DIM
    DATA_DIM = d


def set_outlier_ratio(p):
    global OUTLIER_RATE
    OUTLIER_RATE = p


def compute_auc(labels, scores):
    return roc_auc_score(labels, scores)


def get_percentile(scores, threshold):
    per = np.percentile(scores, 100 - int(100 * threshold))
    return per


def compute_f1_score(labels, scores):
    per = get_percentile(scores, OUTLIER_RATE)
    y_pred = (scores >= per)
    # print(np.sum(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(labels.astype(int),
                                                               y_pred.astype(int),
                                                               average='binary')
    return precision, recall, f1


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
