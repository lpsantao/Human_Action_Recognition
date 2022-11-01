import numpy as np
import pandas as pd
from scipy.signal import medfilt
import sys
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
pd.set_option('display.max_rows', 500)
np.set_printoptions(threshold=sys.maxsize)




class Config():
    def __init__(self):
        self.frame_l = 32   # the length of frames
        self.joint_number = 22  # the number of joints
        self.joint_dim = 3  # dimension of classes
        self.data_dir = 'D:/datasets/HandGestureDataset_SHREC2017/'
        self.save_dir = 'D:/datasets/saved_SHREC/'


C = Config()

train_list = np.loadtxt(C.data_dir + 'train_gestures.txt').astype('int16')
test_list = np.loadtxt(C.data_dir + 'test_gestures.txt').astype('int16')


# Create Dict for training and test sets
Train = {}
Train['pose'] = []
Train['label_14'] = []
Train['label_28'] = []
Test = {}
Test['pose'] = []
Test['label_14'] = []
Test['label_28'] = []


# the file Train_list has info about the train sequences.
# The file has 1960 (70% dataset): id_gesture; id_finger; id_subject; id_essai; 14_labels; 28_labels and size_sequence
for i in tqdm(range(len(train_list))):
    id_gesture = train_list[i][0]
    id_finger = train_list[i][1]
    id_subject = train_list[i][2]
    id_essai = train_list[i][3]
    label_14 = train_list[i][4]
    label_28 = train_list[i][5]

    # got to dir where skeleton data is for each gesture/finger/subject...
    skeleton_path = C.data_dir + '/gesture_' + str(id_gesture) + '/finger_' \
                    + str(id_finger) + '/subject_' + str(id_subject) + '/essai_' + str(id_essai) + '/'

    # load skeletons_world file
    p = np.loadtxt(skeleton_path + 'skeletons_world.txt').astype('float32')
    for j in range(p.shape[1]):
        p[:, j] = medfilt(p[:, j])

    Train['pose'].append(p)
    Train['label_14'].append(label_14)      # coarse labels
    Train['label_28'].append(label_28)      # fine labels

train_df = pd.DataFrame.from_dict(Train)

for i in tqdm(range(len(test_list))):
    id_gesture = test_list[i][0]
    id_finger = test_list[i][1]
    id_subject = test_list[i][2]
    id_essai = test_list[i][3]
    label_14 = test_list[i][4]
    label_28 = test_list[i][5]

    skeleton_path = C.data_dir + '/gesture_' + str(id_gesture) + '/finger_' \
                    + str(id_finger) + '/subject_' + str(id_subject) + '/essai_' + str(id_essai) + '/'

    p = np.loadtxt(skeleton_path + 'skeletons_world.txt').astype('float32')
    for j in range(p.shape[1]):
        p[:, j] = medfilt(p[:, j])

    Test['pose'].append(p)
    Test['label_14'].append(label_14)
    Test['label_28'].append(label_28)


#pickle.dump(Test, open(C.save_dir+"test.pkl", "wb"))
#pickle.dump(Train, open(C.save_dir+"train.pkl", "wb"))