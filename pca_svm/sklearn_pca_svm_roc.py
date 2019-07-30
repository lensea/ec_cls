

import cv2
import numpy as np
from time import time
import logging
import matplotlib.pyplot as plt
from numpy import *
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from itertools import cycle
from sklearn.preprocessing import label_binarize
import pickle
from joblib import dump, load

IMAGE_SIZE = 112
CLS_NUM = 22
train_path = "E:/Data/fov_cls/trains_sub_1.txt"
test_path = "E:/Data/fov_cls/tests.txt"


def svm(trainDataSimplified, trainLabel, testDataSimplified):
    clf3 = SVC(C=11.0)  # C为分类数目
    clf3.fit(trainDataSimplified, trainLabel)
    return clf3.predict(testDataSimplified)

def knn(neighbor, traindata, trainlabel, testdata):
    neigh = KNeighborsClassifier(n_neighbors=neighbor)
    neigh.fit(traindata, trainlabel)
    return neigh.predict(testdata)

if __name__ == '__main__':
   

    fd = open(train_path)
    lines = fd.readlines()
    fd.close()
    num = len(lines)
    data = []
    label = [] 
    for i in range(len(lines)):
        line =lines[i].strip('\n').split(' ')
        #print(line[0])
        #print(line[1])

        img_gray = cv2.imread(line[0],0)
        img_gray = cv2.resize(img_gray,(IMAGE_SIZE,IMAGE_SIZE))
        img_col = img_gray.reshape(IMAGE_SIZE*IMAGE_SIZE)
        data.append(img_col)
        label.append(int(line[1]))
    #print(label)
    #print(data)
    h, w = IMAGE_SIZE,IMAGE_SIZE
    X = np.array(data)  
    print(X.shape)
    y = np.array(label)
    n_samples,n_features = X.shape
    n_classes = len(np.unique(y))
    target_names = []
    for i in range(n_classes):
        names = str(i)
        target_names.append(names)
    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)
    n_components = 128
    print("Extracting the top %d eigenfaces from %d faces"% (n_components, X_train.shape[0]))
    
    #选择一种svd方式,whiten是一种数据预处理方式，会损失一些数据信息，但可获得更好的预测结果
    pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
   
    eigenfaces = pca.components_.reshape((n_components, h, w))#特征脸

    print("Projecting the input data on the eigenfaces orthonormal basis")
    
    X_train_pca = pca.transform(X_train)#得到训练集投影系数
    X_val_pca = pca.transform(X_val)#得到测试集投影系数

    print("Fitting the classifier to the training set")

    '''C为惩罚因子，越大模型对训练数据拟合程度越高，gama是高斯核参数'''
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    '''
    class_weight='balanced'
    表示调整各类别权重，权重与该类中样本数成反比，防止模型过于拟合某个样本数量过大的类
    '''
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',probability=True), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    s = pickle.dumps(clf)
    dump(clf, 'pca_svm.joblib') 
    print("Predicting people's names on the test set")

    ###########  test  ############
    fd = open(test_path)
    lines = fd.readlines()
    fd.close()
    num = len(lines)
    data = []
    label = [] 
    for i in range(len(lines)):
        line =lines[i].strip('\n').split(' ')
        #print(line[0])
        #print(line[1])

        img_gray = cv2.imread(line[0],0)
        img_gray = cv2.resize(img_gray,(IMAGE_SIZE,IMAGE_SIZE))
        img_col = img_gray.reshape(IMAGE_SIZE*IMAGE_SIZE)
        data.append(img_col)
        label.append(int(line[1]))
    #print(label)
    #print(data)
    h, w = IMAGE_SIZE,IMAGE_SIZE
    X = np.array(data)  
    print(X.shape)
    y = np.array(label)
    X_TEST, _, Y_TEST, _ = train_test_split(X, y, test_size=0.00001, random_state=0)
    print(X_TEST.shape)
    print(Y_TEST.shape)
    X_TEST_PCA = pca.transform(X_TEST)#得到测试集投影系数
    t0 = time()
    y_pred = clf.predict(X_TEST_PCA)
    prob = clf.decision_function(X_TEST_PCA)#clf.predict_proba(X_test_pca)
    print("done in %0.3fs" % (time() - t0))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_ = label_binarize(Y_TEST, classes=[0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
    #print(y_)
    scores_=prob
    #print(scores_)
    for i in range(CLS_NUM):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_[:, i], scores_[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])


    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_.ravel(), scores_.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(CLS_NUM)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(CLS_NUM):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= CLS_NUM

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=2)

    colors = cycle(['b','g','r','c','m','y','k','aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(CLS_NUM), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")

    plt.savefig("squeeze_multiclass ROC curve")
    plt.show()

    