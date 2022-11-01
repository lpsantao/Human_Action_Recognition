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
import os
import tensorflow as tf
import seaborn as sns
import keras

np.set_printoptions(threshold=sys.maxsize)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LABELS = ["Grab_2", "Grab", "Expand_2", "Expand", "Pinch_2", "Pinch", "Rot-CW_2", "Rot-CW", "Rot-CCW_2", "Rot-CCW",
          "Tap_2", "Tap", "Swipe-R_2", "Swipe-R", "Swipe-L_2", "Swipe-L", "Swipe-Up_2", "Swipe-Up",
          "Swipe-Dw_2", "Swipe-Dw", "Swipe-X_2", "Swipe-X", "Swipe-V_2", "Swipe-V", "Swipe+_2", "Swipe+", "Shake_2", "Shake"]

def normalize_range(p):
    # normolize to start point, use the center for hand case
    p[:, :, 0] = p[:, :, 0] - np.mean(p[:, :, 0])
    p[:, :, 1] = p[:, :, 1] - np.mean(p[:, :, 1])
    p[:, :, 2] = p[:, :, 2] - np.mean(p[:, :, 2])
    return p

def zoom2(p, target_l=64, joints_num=22, joints_dim=3):
    l = p.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p_new[:, m, n] = medfilt(p_new[:, m, n], 3)
            p_new[:, m, n] = inter.zoom(p[:, m, n], target_l / l)[:target_l]
    return p_new

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(28)
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

class Config():
    def __init__(self):
        self.frame_l = 32  # the length of frames
        self.joint_n = 22  # the number of joints
        self.joint_d = 3  # the dimension of joints
        self.clc_coarse = 14  # the number of coarse class
        self.clc_fine = 28  # the number of fine-grained class
        self.feat_d = 231
        self.filters = 16
        self.data_dir = 'D:/datasets/saved_SHREC/'


C = Config()

Test = pickle.load(open(C.data_dir + "test.pkl", "rb"))
Train = pickle.load(open(C.data_dir + "train.pkl", "rb"))

plt.title('Nº of skeletons per Gesture in Test Set', fontsize=15)
ax = sns.countplot((Test['label_28']), palette="Set1")
ax.set(xlabel='Gesture labels', ylabel='Count')
plt.xticks(rotation=90)
plt.show()

plt.title('Nº of skeletons per Gesture in Train Set', fontsize=15)
ax = sns.countplot(Train['label_28'], palette="Set1")
ax.set(xlabel='Gesture labels', ylabel='Count')
plt.xticks(rotation=90)
plt.show()



# Rescale to be 64 frames : resizing function
def zoom(p, target_l=64, joints_num=66):
    l = p.shape[0]
    p_new = np.empty([target_l, joints_num])
    for m in range(joints_num):
        p_new[:, m] = medfilt(p_new[:, m], 3)
        p_new[:, m] = inter.zoom(p[:, m], target_l / l)[:target_l]
    return p_new


X_test = []
Y_test = []
for i in tqdm(range(len(Test['pose']))):
    p = np.copy(Test['pose'][i]).reshape([-1, 22, 3])
    p = zoom2(p, target_l=64, joints_num=C.joint_n)
    p = normalize_range(p)
    p = p.reshape([-1, 66])
    '''
    padding = np.zeros([200, 66])
    padding[:pp.shape[0], :pp.shape[1]] = pp
    X_1.append(padding)
    X_11= np.array(X_1)
    '''
    X_test.append(p)

    label = np.zeros(C.clc_fine)
    label[Test['label_28'][i] - 1] = 1

    Y_test.append(label)

X_test = np.stack(X_test)
Y_test = np.stack(Y_test)

X_train = []
Y_train = []
for i in tqdm(range(len(Train['pose']))):
    p_tr = np.copy(Train['pose'][i]).reshape([-1, 22, 3])
    p_tr = zoom2(p_tr, target_l=64, joints_num=C.joint_n)
    p_tr = normalize_range(p_tr)
    p_tr = p_tr.reshape([-1, 66])

    X_train.append(p_tr)

    label_train = np.zeros(C.clc_fine)
    label_train[Train['label_28'][i] - 1] = 1

    Y_train.append(label_train)

X_train = np.stack(X_train)
Y_train = np.stack(Y_train)

verbose = 1
epochs = 100
batch_size = 64
plot = True

n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]
print(n_outputs)
'''
model = keras.Sequential()
model.add(keras.layers.LSTM(100, input_shape=(n_timesteps, n_features), return_sequences=True))
model.add(keras.layers.LSTM(100, input_shape=(n_timesteps, n_features)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(n_outputs, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    Y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, Y_test),
                    verbose=verbose)

_, accuracy = model.evaluate(X_test,
                             Y_test,
                             batch_size=batch_size,
                             verbose=verbose)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

model.save('lstm_model_28_normalized.h5')
'''
# load model from single file
model = load_model('lstm_model_28_normalized.h5')

# make predictions
y_classes = model.predict_classes(X_test)
y_classes = [x+1 for x in y_classes]
print("")
print("Precision: {}%".format(100 * metrics.precision_score(Test['label_28'], y_classes, average="weighted")))
print("Recall: {}%".format(100 * metrics.recall_score(Test['label_28'], y_classes, average="weighted")))
print("f1_score: {}%".format(100 * metrics.f1_score(Test['label_28'], y_classes, average="weighted")))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(Test['label_28'], y_classes)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)

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
tick_marks = np.arange((28))
plt.xticks(tick_marks, LABELS, rotation='vertical')
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylim([27, -.5])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.figure(figsize=(12,13))
plot_confusion_matrix(confusion_matrix, classes=LABELS,
                      normalize=True, title='Normalized confusion matrix', cmap=plt.cm.rainbow)
plt.ylim([27, -.5])
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
a =multiclass_roc_auc_score(Test['label_28'], y_classes)
print(a)



if plot:
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

