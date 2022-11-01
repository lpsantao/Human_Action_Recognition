import itertools
from tqdm import tqdm
import pickle
import sys
import numpy as np
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
from scipy.spatial.distance import cdist
import os
import tensorflow as tf
import seaborn as sns
import keras

np.set_printoptions(threshold=sys.maxsize)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
LABELS = ["Grab", "Expand", "Pinch", "Rot-CW" , "Rot-CCW", "Tap", "Swipe-R", "Swipe-L", "Swipe-Up",
          "Swipe-Dw", "Swipe-X","Swipe-V","Swipe+", "Shake"]

class Config():
    def __init__(self):
        self.frame_l = 64  # the length of frames
        self.joint_n = 22  # the number of joints
        self.joint_d = 3  # the dimension of joints
        self.clc_coarse = 14  # the number of coarse class
        self.clc_fine = 28  # the number of fine-grained class
        self.feat_d = 231
        self.filters = 16
        self.data_dir = 'D:/datasets/saved_SHREC/'


C = Config()
xx=[]
Test = pickle.load(open(C.data_dir + "test.pkl", "rb"))
Train = pickle.load(open(C.data_dir + "train.pkl", "rb"))   #[file][frame][22*3-xyz, joints]

'''
plt.title('Nº of skeletons per Gesture in Test Set', fontsize=15)
ax = sns.countplot((Test['label_14']), palette="Set1")
ax.set(xlabel='Gesture labels', ylabel='Count')
plt.xticks(rotation=90)
plt.show()

plt.title('Nº of skeletons per Gesture in Train Set', fontsize=15)
ax = sns.countplot(Train['label_14'], palette="Set1")
ax.set(xlabel='Gesture labels', ylabel='Count')
plt.xticks(rotation=90)
plt.show()
'''
# Calculate JCD feature
def get_CG(p, C):
    M = []
    iu = np.triu_indices(C.joint_n, 1, C.joint_n)
    for f in range(C.frame_l):
        # distance max
        d_m = cdist(p[f], np.concatenate([p[f], np.zeros([1, C.joint_d])]), 'euclidean')
        d_m = d_m[iu]
        M.append(d_m)
    M = np.stack(M)
    return M


def normalize_range(p):
    # normolize to start point, use the center for hand case
    p[:, :, 0] = p[:, :, 0] - np.mean(p[:, :, 0])
    p[:, :, 1] = p[:, :, 1] - np.mean(p[:, :, 1])
    p[:, :, 2] = p[:, :, 2] - np.mean(p[:, :, 2])
    return p


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(14)
    plt.xticks(tick_marks, classes, rotation='vertical')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Rescale to be 64 frames : resizing function
def zoom(p, target_l=200, joints_num=66):
    l = p.shape[0]
    p_new = np.empty([target_l, joints_num])
    for m in range(joints_num):
        p_new[:, m] = medfilt(p_new[:, m], 3)
        p_new[:, m] = inter.zoom(p[:, m], target_l / l)[:target_l]
    return p_new


def zoom2(p, target_l=200, joints_num=22, joints_dim=3):
    l = p.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p_new[:, m, n] = medfilt(p_new[:, m, n], 3)
            p_new[:, m, n] = inter.zoom(p[:, m, n], target_l / l)[:target_l]
    return p_new

X_test = []
X_0 = []
Y_test = []
dumm = []
verbose = 1
epochs = 100
batch_size = 64
plot = True
p=[]
#for i in tqdm(range(len(Train['pose']))):
#    p2 = np.copy(Train['pose'][i]).reshape([-1, 22, 3])
#    p.append(p2)
# x1=[]
# x2 =[]
# x3= []
# x4=[]
# x5=[]
# x6= []
# x7= []
# x8= []
# x9= []
# x10= []
# x11= []
# x12= []
# x13=[]
# x14 = []
# y1 =[]
# y4 =[]
# y3= []
# y6=[]
# y5=[]
# y7= []
# y2= []
# y8= []
# y9= []
# y10= []
# y11= []
# y12= []
# y13=[]
# y14 = []
# z1 =[]
# z4 =[]
# z3= []
# z6=[]
# z5=[]
# z7= []
# z2= []
# z8= []
# z9= []
# z10= []
# z11= []
# z12= []
# z13=[]
# z14 = []
# for i in tqdm(range(len(p))):
#     if Train['label_14'][i] == 1:
#         aa=len(p[i])
#         for j in range(aa):
#             x1.append(p[i][j][9][0])
#             y1.append(p[i][j][9][1])
#             z1.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 2:
#         aa=len(p[i])
#         for j in range(aa):
#             x2.append(p[i][j][9][0])
#             y2.append(p[i][j][9][1])
#             z2.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 3:
#         aa=len(p[i])
#         for j in range(aa):
#             x3.append(p[i][j][9][0])
#             y3.append(p[i][j][9][1])
#             z3.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 4:
#         aa=len(p[i])
#         for j in range(aa):
#             x4.append(p[i][j][9][0])
#             y4.append(p[i][j][9][1])
#             z4.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 5:
#         aa=len(p[i])
#         for j in range(aa):
#             x5.append(p[i][j][9][0])
#             y5.append(p[i][j][9][1])
#             z5.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 6:
#         aa=len(p[i])
#         for j in range(aa):
#             x6.append(p[i][j][9][0])
#             y6.append(p[i][j][9][1])
#             z6.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 7:
#         aa=len(p[i])
#         for j in range(aa):
#             x7.append(p[i][j][9][0])
#             y7.append(p[i][j][9][1])
#             z7.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 8:
#         aa=len(p[i])
#         for j in range(aa):
#             x8.append(p[i][j][9][0])
#             y8.append(p[i][j][9][1])
#             z8.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 9:
#         aa=len(p[i])
#         for j in range(aa):
#             x9.append(p[i][j][9][0])
#             y9.append(p[i][j][9][1])
#             z9.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 10:
#         aa=len(p[i])
#         for j in range(aa):
#             x10.append(p[i][j][9][0])
#             y10.append(p[i][j][9][1])
#             z10.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 11:
#         aa=len(p[i])
#         for j in range(aa):
#             x11.append(p[i][j][9][0])
#             y11.append(p[i][j][9][1])
#             z11.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 12:
#         aa=len(p[i])
#         for j in range(aa):
#             x12.append(p[i][j][9][0])
#             y12.append(p[i][j][9][1])
#             z12.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 13:
#         aa=len(p[i])
#         for j in range(aa):
#             x13.append(p[i][j][9][0])
#             y13.append(p[i][j][9][1])
#             z13.append(p[i][j][9][2])
#     elif Train['label_14'][i] == 14:
#         aa=len(p[i])
#         for j in range(aa):
#             x14.append(p[i][j][9][0])
#             y14.append(p[i][j][9][1])
#             z14.append(p[i][j][9][2])
#
#
# plt.figure(figsize=(16,10))
# plt.subplot(2,3,1)
# plt.title('X values distribution per activity (joint6)')
# sns.distplot(x1,hist = False, label = 'grab')
# sns.distplot(x2,hist = False,label = 'expand')
# sns.distplot(x3,hist = False, label = 'pinch')
# sns.distplot(x4,hist = False, label = 'rot-CW')
# sns.distplot(x5,hist = False,label = 'rot-CCW')
# sns.distplot(x6,hist = False,label = 'tap')
# sns.distplot(x7,hist = False, label = 'swipe-R')
# sns.color_palette('bright')
# plt.legend(loc='left')
# plt.axis([-0.25, 1.25, 0, 5])
# plt.subplot(2,3,4)
# sns.distplot(x8,hist = False, label = 'swipe-L')
# sns.distplot(x9,hist = False,label = 'swipe-up')
# sns.distplot(x10,hist = False, label = 'swipe-dw')
# sns.distplot(x11,hist = False, label = 'swipe-x')
# sns.distplot(x12,hist = False,label = 'swipe-v')
# sns.distplot(x13,hist = False, label = 'swipe+')
# sns.distplot(x14,hist = False, label = 'shake')
# sns.color_palette('bright')
# plt.legend(loc='left')
# plt.axis([-0.25, 1.25, 0, 5])
# plt.subplot(2,3,2)
# plt.title('Y values distribution per activity')
# sns.distplot(y1,hist = False, label = 'grab')
# sns.distplot(y2,hist = False,label = 'expand')
# sns.distplot(y3,hist = False, label = 'pinch')
# sns.distplot(y4,hist = False, label = 'rot-CW')
# sns.distplot(y5,hist = False,label = 'rot-CCW')
# sns.distplot(y6,hist = False,label = 'tap')
# sns.distplot(y7,hist = False, label = 'swipe-R')
# sns.color_palette('bright')
# plt.legend(loc='left')
# plt.axis([-.75, .35, 0, 12])
# plt.subplot(2,3,5)
# sns.distplot(y8,hist = False, label = 'swipe-L')
# sns.distplot(y9,hist = False,label = 'swipe-up')
# sns.distplot(y10,hist = False, label = 'swipe-dw')
# sns.distplot(y11,hist = False, label = 'swipe-x')
# sns.distplot(y12,hist = False,label = 'swipe-v')
# sns.distplot(y13,hist = False, label = 'swipe+')
# sns.distplot(y14,hist = False, label = 'shake')
# sns.color_palette('bright')
# plt.axis([-.75, .25, 0, 12])
# plt.tight_layout()
# plt.legend(loc='left')
# plt.subplot(2,3,3)
# plt.title('Z values distribution per activity')
# sns.distplot(z1,hist = False, label = 'grab')
# sns.distplot(z2,hist = False,label = 'expand')
# sns.distplot(z3,hist = False, label = 'pinch')
# sns.distplot(z4,hist = False, label = 'rot-CW')
# sns.distplot(z5,hist = False,label = 'rot-CCW')
# sns.distplot(z6,hist = False,label = 'tap')
# sns.distplot(z7,hist = False, label = 'swipe-R')
# sns.color_palette('bright')
# plt.legend(loc='left')
# plt.axis([0, 1, 0, 7])
# plt.subplot(2,3,6)
# sns.distplot(z8,hist = False, label = 'swipe-L')
# sns.distplot(z9,hist = False,label = 'swipe-up')
# sns.distplot(z10,hist = False, label = 'swipe-dw')
# sns.distplot(z11,hist = False, label = 'swipe-x')
# sns.distplot(z12,hist = False,label = 'swipe-v')
# sns.distplot(z13,hist = False, label = 'swipe+')
# sns.distplot(z14,hist = False, label = 'shake')
# sns.color_palette('bright')
# plt.axis([0, 1, 0, 7])
# plt.tight_layout()
# plt.legend(loc='left')
# plt.show()




for i in tqdm(range(len(Test['pose']))):
    p = np.copy(Test['pose'][i]).reshape([-1, 22, 3])
    p = zoom2(p, target_l=64, joints_num=C.joint_n)
    p = normalize_range(p)
    pp = p.reshape([-1, 66])

    '''
    pp = np.copy(Test['pose'][i])  # [n_frames, 22*3] 22 joints, x,y,z
    print(pp.shape)
    p = zoom(pp, target_l=64, joints_num=66)
    print(p.shape)
    p = normalize_range(p)
    '''
    #padding = np.zeros([200, 66])
    #padding[:pp.shape[0], :pp.shape[1]] = pp
    #X_1.append(padding)
    #X_11= np.array(X_1)
    
    X_test.append(pp)

    label = np.zeros(C.clc_coarse)
    label[Test['label_14'][i] - 1] = 1


    Y_test.append(label)
    M = get_CG(p, C)

    X_0.append(M)

X_0 = np.stack(X_0)
X_test = np.stack(X_test)
Y_test = np.stack(Y_test)

X_1 = []
X_train = []
Y_train = []
for i in tqdm(range(len(Train['pose']))):
    p_tr = np.copy(Train['pose'][i]).reshape([-1, 22, 3])
    p_tr = zoom2(p_tr, target_l=64, joints_num=C.joint_n)
    p_tr = normalize_range(p_tr)
    p_tr2 = p_tr.reshape([-1, 66])

    #pp_tr = np.copy(Train['pose'][i])  # [n_frames, 22*3] 22 joints, x,y,z
    #p_tr = zoom(pp_tr, target_l=64, joints_num=66)

    X_train.append(p_tr2)

    label_train = np.zeros(C.clc_coarse)
    label_train[Train['label_14'][i] - 1] = 1

    Y_train.append(label_train)
    M = get_CG(p_tr, C)


    X_1.append(M)

X_1 = np.stack(X_1)
X_train = np.stack(X_train)
print(X_train.shape)
print(X_1.shape)
Y_train = np.stack(Y_train)

n_timesteps, n_features, n_outputs = X_train.shape[1], X_1.shape[2], Y_train.shape[1]

model = keras.Sequential()
model.add(keras.layers.LSTM(100, input_shape=(n_timesteps, n_features), return_sequences=True))
model.add(keras.layers.LSTM(100, input_shape=(n_timesteps, n_features)))
model.add(keras.layers.Dropout(0.7))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(n_outputs, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_1,
                    Y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_0, Y_test),
                    verbose=verbose)

_, accuracy = model.evaluate(X_0,
                             Y_test,
                             batch_size=batch_size,
                             verbose=verbose)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
model.save('lstm_model_normalized_feature_84batch.h5')


# load model from single file
model = load_model('lstm_model_normalized.h5')

# make predictions
y_classes = model.predict_classes(X_test)
y_classes = [x+1 for x in y_classes]
print("")
print("Precision: {}%".format(100 * metrics.precision_score(Test['label_14'], y_classes, average="weighted")))
print("Recall: {}%".format(100 * metrics.recall_score(Test['label_14'], y_classes, average="weighted")))
print("f1_score: {}%".format(100 * metrics.f1_score(Test['label_14'], y_classes, average="weighted")))

#print("")
#print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(Test['label_14'], y_classes)
#print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100

#print("")
#print("Confusion matrix (normalised to % of total test data):")
#print(normalised_confusion_matrix)

# Plot Results:
width = 12
height = 13
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange((14))
plt.xticks(tick_marks, LABELS, rotation='vertical')
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylim([13.5, -.5])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.figure(figsize=(12,13))
plot_confusion_matrix(confusion_matrix, classes=LABELS,
                      normalize=True, title='Normalized confusion matrix', cmap=plt.cm.rainbow)
plt.ylim([13.5, -.5])
plt.tight_layout()
plt.show()

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    fig, c_ax = plt.subplots()
    for (idx, c_label) in enumerate(LABELS): # all_labels: no of the labels
        fpr, tpr, thresholds = metrics.roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label ='%s (AUC: %0.2f)' % (c_label, metrics.auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b--', label = 'Random Guessing')
    plt.legend(loc='best')
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.show()
    return metrics.roc_auc_score(y_test, y_pred, average=average)

# calling
a =multiclass_roc_auc_score(Test['label_14'], y_classes)
print(a)


if plot:
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
