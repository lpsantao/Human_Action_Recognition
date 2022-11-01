import matplotlib.pyplot as plt
from jhmd_utils import *
from tqdm import tqdm
import pickle
import seaborn as sns
from sklearn import preprocessing
import itertools
from keras.layers.core import *
from sklearn import metrics
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import keras

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


random.seed(123)

LABEL = ['clap', 'pour', 'run', 'wave', 'throw', 'golf', 'stand', 'sit', 'jump', 'pullup', 'pick',
         'walk', 'push', 'catch']


# Rescale to be 64 frames : resizing function
def zoom2(p, target_l=128, joints_num=30):
    l = p.shape[0]
    p_new = np.empty([target_l, joints_num])
    for m in range(joints_num):
        p_new[:, m] = medfilt(p_new[:, m], 3)
        p_new[:, m] = inter.zoom(p[:, m], target_l / l)[:target_l]
    return p_new


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    fig, c_ax = plt.subplots()
    for (idx, c_label) in enumerate(LABEL): # all_labels: no of the labels
        fpr, tpr, thresholds = metrics.roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label ='%s (AUC: %0.2f)' % (c_label, metrics.auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b--', label = 'Random Guessing')
    plt.legend(loc='best')
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.show()
    return metrics.roc_auc_score(y_test, y_pred, average=average)

class Config():
    def __init__(self):
        self.frame_l = 128 # the length of frames
        self.joint_n = 15 # the number of joints
        self.joint_d = 2 # the dimension of joints
        self.clc_num = 14 # the number of class
        self.feat_d = 105
        self.filters = 16
        self.save_dir = 'D:\\datasets\\saved_JHMDB\\'
C = Config()


def data_generator(T,C,le):
    X_0 = []
    X_1 = []
    Y = []
    for i in tqdm(range(len(T['pose']))):
        p = np.copy(T['pose'][i]).reshape([-1,30])
        p = zoom2(p, target_l=C.frame_l, joints_num=30)
        #p = norm_scale(p)

        label = np.zeros(C.clc_num)
        label[le.transform(T['label'])[i]-1] = 1

        #M = get_CG(p2,C)

        #X_0.append(M)
        X_1.append(p)
        Y.append(label)

    #X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    Y = np.stack(Y)
    return X_1,Y

import matplotlib.cm as cm

Train = pickle.load(open(C.save_dir+"GT_train_1.pkl", "rb"))
Train2 = pickle.load(open(C.save_dir+"GT_train_2.pkl", "rb"))
Train3 = pickle.load(open(C.save_dir+"GT_train_3.pkl", "rb"))
Test = pickle.load(open(C.save_dir+"GT_test_1.pkl", "rb"))
Test2 = pickle.load(open(C.save_dir+"GT_test_2.pkl", "rb"))
Test3 = pickle.load(open(C.save_dir+"GT_test_3.pkl", "rb"))

xclap =[]
xstand =[]
xgolf= []
xpour=[]
xsit=[]
xjump= []
xwave= []
xpull= []
xthrow= []
xpick= []
xwalk= []
xcatch= []
xrun=[]
xpush = []
yclap =[]
ystand =[]
ygolf= []
ypour=[]
ysit=[]
yjump= []
ywave= []
ypull= []
ythrow= []
ypick= []
ywalk= []
ycatch= []
yrun=[]
ypush = []

for i in range(433):
    print(Train['label'][i])
    if Train['label'][i] == 'clap':
        aa=len(Train['pose'][i])
        print(aa)
        for j in range(aa):
            for kk in range (15):
                xclap.append(Train['pose'][i][j][kk][0])
                yclap.append(Train['pose'][i][j][kk][1])
            plt.plot(yclap, xclap, 'o')
        plt.show()
        xclap=[]
        yclap = []
    elif Train['label'][i] == 'wave':
        aa=len(Train['pose'][i])
        for j in range(aa):
            for kk in range(15):
                xwave.append(Train['pose'][i][j][kk][0])
                ywave.append(Train['pose'][i][j][kk][1])
        plt.scatter(ywave, xwave)
        plt.show()
        xwave=[]
        ywave = []
    elif Train['label'][i] == 'throw':
        aa=len(Train['pose'][i])
        for j in range(aa):
            for kk in range(15):
                xthrow.append(Train['pose'][i][j][kk][0])
                ythrow.append(Train['pose'][i][j][kk][1])
        plt.scatter(ythrow, xthrow)
        plt.show()
        xthrow=[]
        ythrow = []
    elif Train['label'][i] == 'stand':
        aa=len(Train['pose'][i])
        for j in range(aa):
            for kk in range(15):
                xstand.append(Train['pose'][i][j][kk][0])
                ystand.append(Train['pose'][i][j][kk][1])
        plt.scatter(ystand, xstand)
        plt.show()
        xstand=[]
        ystand = []
    elif Train['label'][i] == 'jump':
        aa = len(Train['pose'][i])
        for j in range(aa):
            for k in range(15):
                xjump.append(Train['pose'][i][j][k][0])
                yjump.append(Train['pose'][i][j][k][1])
        plt.scatter(yjump, xjump)
        plt.show()
        xjump=[]
        yjump = []
    elif Train['label'][i] == 'walk':
        aa = len(Train['pose'][i])
        for j in range(aa):
            for k in range(15):
                xwalk.append(Train['pose'][i][j][k][0])
                ywalk.append(Train['pose'][i][j][k][1])
        plt.scatter(ywalk, xwalk)
        plt.show()
        xwalk=[]
        ywalk = []
    elif Train['label'][i] == 'push':
        aa=len(Train['pose'][i])
        for j in range(aa):
            for k in range(15):
                xpush.append(Train['pose'][i][j][k][0])
                ypush.append(Train['pose'][i][j][k][1])
        plt.scatter(ypush, xpush)
        plt.show()
        xpush =[]
        ypush =[]
    '''
    elif Train['label'][i] == 'golf':
        aa=len(Train['pose'][i])
        for j in range(aa):
            xgolf.append(Train['pose'][i][j][5][0])
            ygolf.append(Train['pose'][i][j][5][1])
    elif Train['label'][i] == 'sit':
        aa=len(Train['pose'][i])
        for j in range(aa):
            xsit.append(Train['pose'][i][j][5][0])
            ysit.append(Train['pose'][i][j][5][1])
    elif Train['label'][i] == 'jump':
        aa=len(Train['pose'][i])
        for j in range(aa):
            xjump.append(Train['pose'][i][j][5][0])
            yjump.append(Train['pose'][i][j][5][1])
    elif Train['label'][i] == 'pullup':
        aa=len(Train['pose'][i])
        for j in range(aa):
            xpull.append(Train['pose'][i][j][5][0])
            ypull.append(Train['pose'][i][j][5][1])
    elif Train['label'][i] == 'pick':
        aa=len(Train['pose'][i])
        for j in range(aa):
            xpick.append(Train['pose'][i][j][5][0])
            ypick.append(Train['pose'][i][j][5][1])
    elif Train['label'][i] == 'walk':
        aa=len(Train['pose'][i])
        for j in range(aa):
            xwalk.append(Train['pose'][i][j][5][0])
            ywalk.append(Train['pose'][i][j][5][1])
    elif Train['label'][i] == 'catch':
        aa=len(Train['pose'][i])
        for j in range(aa):
            xcatch.append(Train['pose'][i][j][5][0])
            ycatch.append(Train['pose'][i][j][5][1])
    elif Train['label'][i] == 'run':
        aa=len(Train['pose'][i])
        for j in range(aa):
            xrun.append(Train['pose'][i][j][5][0])
            yrun.append(Train['pose'][i][j][5][1])
    elif Train['label'][i] == 'push':
        aa=len(Train['pose'][i])
        for j in range(aa):
            xpush.append(Train['pose'][i][j][5][0])
            ypush.append(Train['pose'][i][j][5][1])
    elif Train['label'][i] == 'pour':
        aa=len(Train['pose'][i])
        for j in range(aa):
            xpour.append(Train['pose'][i][j][5][0])
            ypour.append(Train['pose'][i][j][5][1])

plt.figure(figsize=(16,10))
plt.subplot(2,2,1)
plt.title('X values distribution per activity (joint6)')
sns.distplot(xclap,hist = False, label = 'Clap')
sns.distplot(xpour,hist = False,label = 'Pour')
sns.distplot(xrun,hist = False, label = 'Run')
sns.distplot(xwave,hist = False, label = 'Wave')
sns.distplot(xthrow,hist = False,label = 'Throw')
sns.distplot(xgolf,hist = False,label = 'Golf')
sns.distplot(xstand,hist = False, label = 'Stand')
sns.color_palette('bright')
plt.legend(loc='left')
plt.axis([-2, 2, 0, 10])
plt.subplot(2,2,3)
sns.distplot(xsit,hist = False, label = 'Sit')
sns.distplot(xjump,hist = False,label = 'Jump')
sns.distplot(xpull,hist = False, label = 'Pullup')
sns.distplot(xpick,hist = False, label = 'Pick')
sns.distplot(xwalk,hist = False,label = 'Walk')
sns.distplot(xpush,hist = False, label = 'Push')
sns.distplot(xcatch,hist = False, label = 'Catch')
sns.color_palette('bright')
plt.legend(loc='left')
plt.axis([-2, 2, 0, 10])
plt.subplot(2,2,2)
plt.title('Y values distribution per activity')
sns.distplot(yclap,hist = False, label = 'Clap')
sns.distplot(ypour,hist = False,label = 'Pour')
sns.distplot(yrun,hist = False, label = 'Run')
sns.distplot(ywave,hist = False, label = 'Wave')
sns.distplot(ythrow,hist = False,label = 'Throw')
sns.distplot(ygolf,hist = False,label = 'Golf')
sns.distplot(ystand,hist = False, label = 'Stand')
sns.color_palette('bright')
plt.legend(loc='left')
plt.axis([-2, 2, 0, 10])
plt.subplot(2,2,4)
sns.distplot(ysit,hist = False, label = 'Sit')
sns.distplot(yjump,hist = False,label = 'Jump')
sns.distplot(ypull,hist = False, label = 'Pullup')
sns.distplot(ypick,hist = False, label = 'Pick')
sns.distplot(ywalk,hist = False,label = 'Walk')
sns.distplot(ypush,hist = False, label = 'Push')
sns.distplot(ycatch,hist = False, label = 'Catch')
sns.color_palette('bright')
plt.axis([-2, 2, 0, 10])
plt.tight_layout()
plt.legend(loc='left')
plt.show()
'''
le = preprocessing.LabelEncoder()
le.fit(Train['label'])
X_1,Y = data_generator(Train, C, le)
X_test_1, Y_test = data_generator(Test, C, le)
le2 = preprocessing.LabelEncoder()
le2.fit(Train2['label'])
X_1_split2, Y_split2 = data_generator(Train2, C, le2)
X_test_1_split2, Y_test_split2 = data_generator(Test2, C, le2)
le3 = preprocessing.LabelEncoder()
le3.fit(Train3['label'])
X_1_split3, Y_split3 = data_generator(Train3, C, le3)
X_test_1_split3, Y_test_split3 = data_generator(Test3, C, le3)

Y = np.concatenate((Y,Y_split2))
Y = np.concatenate((Y,Y_split3))
Y_test = np.concatenate((Y_test, Y_test_split2))
Y_test = np.concatenate((Y_test, Y_test_split3))
X_test_1 = np.concatenate((X_test_1,X_test_1_split2))
X_test_1 = np.concatenate((X_test_1,X_test_1_split3))
X_1 = np.concatenate((X_1,X_1_split2))
X_1 = np.concatenate((X_1,X_1_split3))

sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(X_1, hue='ActivityName', size=6,aspect=2)
facetgrid.map(sns.distplot,'tBodyAccMag_mean', hist=False)
plt.show()

n_timesteps, n_features, n_outputs = X_1.shape[1], X_1.shape[2], Y.shape[1]

verbose = 1
epochs = 150
batch_size = 64


model = keras.Sequential()
#model.add(keras.layers.LSTM(100, input_shape=(n_timesteps, n_features), return_sequences=True))
model.add(keras.layers.LSTM(100, input_shape=(n_timesteps, n_features)))
model.add(keras.layers.Dropout(0.6))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(n_outputs, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_1,
                    Y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test_1, Y_test),
                    verbose=verbose)

_, accuracy = model.evaluate(X_test_1,
                             Y_test,
                             batch_size=batch_size,
                             verbose=verbose)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
model.save('lstm_model_JHMD.h5')
'''


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

# load model from single file
model = load_model('lstm_model_JHMD.h5')

# make predictions
y_classes = model.predict_classes(X_test_1)
y_t = [np.where(r==1)[0][0] for r in Y_test]
print("")
print("Precision: {}%".format(100 * metrics.precision_score(y_t, y_classes, average="weighted")))
print("Recall: {}%".format(100 * metrics.recall_score(y_t, y_classes, average="weighted")))
print("f1_score: {}%".format(100 * metrics.f1_score(y_t, y_classes, average="weighted")))

#print("")
#print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_t, y_classes)
#print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100

#print("")
#print("Confusion matrix (normalised to % of total test data):")
#print(normalised_confusion_matrix)

# Plot Results:
width = 12
height = 13
tick_marks = np.arange((14))
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
plt.xticks(tick_marks, LABEL, rotation='vertical')
plt.yticks(tick_marks, LABEL)
plt.tight_layout()
plt.ylim([13.5, -.5])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.figure(figsize=(11,13))
plot_confusion_matrix(confusion_matrix, classes=LABEL,
                      normalize=True, title='Normalized confusion matrix', cmap=plt.cm.rainbow)
plt.xticks(tick_marks, LABEL, rotation='vertical')
plt.yticks(tick_marks, LABEL)
plt.tight_layout()
plt.ylim([13.5, -.5])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# calling
a = multiclass_roc_auc_score(y_t, y_classes)
print(a)

'''

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
'''