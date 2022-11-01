import numpy as np
import scipy.io
import pickle
import glob
import matplotlib.pyplot as plt
from scipy.signal import medfilt


class Config():
    def __init__(self):
        self.frame_l = 32   # the length of frames
        self.joint_n = 15   # the number of joints
        self.joint_d = 2    # the dimension of classes
        self.data_dir = 'D:\\datasets\\JHMDB\\'
        self.save_dir = 'D:\\datasets\\saved_JHMDB\\'


def generate_list(lists):
    train_list = []
    test_list = []
    for i in range(len(lists)):
        lines = []
        with open(lists[i]) as f:      # returns files in list of splits as object (ex: catch_test_split1.txt as object)
            lines.append(f.read().splitlines())     # reads file & splits lines(ex: Frisbee_catch_f_cm_np1_ri_med_0.avi2
        f.close()
        lines = np.hstack(lines)
        for line in lines:
            file_name, flag = line.split(' ')
            if flag == '1':         # if name ends in 1 is train
                train_list.append(file_name.split('.')[0])      # drop the ".avi"
            elif flag == '2':       # if ends in 2 is test
                test_list.append(file_name.split('.')[0])
    return train_list, test_list


def generate_label(joint_list, train_list, test_list):
    train = {}
    train['pose'] = []
    train['label'] = []
    test = {}
    test['pose'] = []
    test['label'] = []
    for i in range(len(joint_list)):
        label = joint_list[i].split('\\')[-2]       # where the label is (ex: clap)
        pose_path = joint_list[i]+'/joint_positions.mat'
        mat = scipy.io.loadmat(pose_path)       # dictionary with variable names as keys, and loaded matrices as values.
        pose = np.round(mat['pos_world'], 4).swapaxes(0, 2)     # swap (x,y) for number of frames
        file = joint_list[i].split('\\')[-1]
        if file in train_list:
            train['label'].append(label)
            train['pose'].append(pose)
        elif file in test_list:
            test['label'].append(label)
            test['pose'].append(pose)
    return train, test


C = Config()
split_lists = glob.glob(C.data_dir + 'splits/*.txt')        # returns list of parameters that matches name
joints_list = glob.glob(C.data_dir + 'joint_positions/*/*')

lists_1 = []
lists_2 = []
lists_3 = []
for file in split_lists:
    if file.split('\\')[-1].split('.')[0].split('_')[-1] == 'split1':
        lists_1.append(file)
    elif file.split('\\')[-1].split('.')[0].split('_')[-1] == 'split2':
        lists_2.append(file)
    elif file.split('\\')[-1].split('.')[0].split('_')[-1] == 'split3':
        lists_3.append(file)


train_list_1, test_list_1 = generate_list(lists_1)
train_1, test_1 = generate_label(joints_list, train_list_1, test_list_1)
train_list_2, test_list_2 = generate_list(lists_2)
train_2, test_2 = generate_label(joints_list, train_list_2, test_list_2)
train_list_3, test_list_3 = generate_list(lists_3)
train_3, test_3 = generate_label(joints_list, train_list_3, test_list_3)

pickle.dump(train_1, open(C.save_dir+"train_1.pkl", "wb"))
pickle.dump(test_1, open(C.save_dir+"test_1.pkl", "wb"))
pickle.dump(train_2, open(C.save_dir+"train_2.pkl", "wb"))
pickle.dump(test_2, open(C.save_dir+"test_2.pkl", "wb"))
pickle.dump(train_3, open(C.save_dir+"train_3.pkl", "wb"))
pickle.dump(test_3, open(C.save_dir+"test_3.pkl", "wb"))