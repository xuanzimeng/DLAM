import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc
from pytorchtools import EarlyStopping
import torch.nn.functional as F
import xlrd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_curve,average_precision_score
from MLP import mlp
from DeepEP import Deepep
#导入数据集samples_data[:,0:96]
sample1 = pd.read_excel(r'E:\testsample.xls')
sample2 = pd.read_excel(r'E:\trainsample.xls')
samples_all=sample1.append(sample2)
# print(samples_all)
#划分数据集为测试集,训练集,验证集
samples_data=samples_all.values
# print(samples_data.shape)
X_data=samples_data[:,0:96]
Y_labels=samples_data[:,96]
X_other,X_test,Y_other,Y_test=train_test_split(X_data,Y_labels,test_size=0.2,random_state=1,stratify=Y_labels)
# print(X_test.shape)
# print(X_other.shape)
# print(Y_test.sum())
# print(Y_other.sum())
#reshape 标签特征
Y_test=Y_test.reshape((Y_test.shape[0],-1))
# print(Y_test.shape)
test_data=np.hstack((X_test,Y_test))
# print(test_data)
# print(test_data.shape)
X_train,X_validate,Y_train,Y_validate=train_test_split(X_other,Y_other,test_size=0.25,random_state=1,stratify=Y_other)
# print(X_train.shape)
# print(X_validate.shape)
# print(Y_validate.sum())
# print(Y_train.sum())
Y_train=Y_train.reshape((Y_train.shape[0],-1))
Y_validate=Y_validate.reshape((Y_validate.shape[0],-1))
train_data=np.hstack((X_train,Y_train))
validate_data=np.hstack((X_validate,Y_validate))

# 重写Pytorch Dataset 类里面的 init len getitem 方法
class DatasetFromNumPy(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)  # 返回长度

    def __getitem__(self, idx):
        # 这里可以对数据进行处理,比如讲字符数值化
        data = self.data[idx]  # 索引到第idx行的数据
        features1 = data[0:36]  # 前四项为特征数据
        features2 = data[36:72]
        features3 = data[72:84]
        features4 = data[84:96]
        # features=features.reshape(1,8,12)
        features1 = torch.from_numpy(features1)
        features2 = torch.from_numpy(features2)
        features3 = torch.from_numpy(features3)
        features4 = torch.from_numpy(features4)
        features1= torch.unsqueeze(features1, dim=0)
        features2=torch.unsqueeze(features2,dim=0)
        features3=torch.unsqueeze(features3,dim=0)
        features4=torch.unsqueeze(features4,dim=0)
        features1 = features1.to(torch.float32)
        features2 = features2.to(torch.float32)
        features3 = features3.to(torch.float32)
        features4 = features4.to(torch.float32)
        # pad = nn.ZeroPad2d(padding=(0, 0, 0, 1))
        # features=pad(features1)
        label = data[96]  # 最后一项为指标数据
        label = torch.from_numpy(np.asarray(label))
        return features1, features2, features3, features4, label  # 返回特征和指标
# DatasetFromNumPy 实例化，生成一个 trainLoader和test_loader，以便于数据处理
train_dataset = DatasetFromNumPy(train_data)
print(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=70, shuffle=True, drop_last=False)  # batch_size为每次读取的样本数量，shuffle是否选择随机读取数据
test_dataset=DatasetFromNumPy(test_data)
test_loader=DataLoader(test_dataset,batch_size=70,shuffle=True,drop_last=False)
validate_dataset=DatasetFromNumPy(validate_data)
validate_loader=DataLoader(validate_dataset,batch_size=1019,shuffle=False,drop_last=False)
# for data in train_loader:
#     f1,f2,f3,f4,la=data
#     print(f1.shape)
#     print(f2.shape)
#     print(f3.shape)
#     print(f4.shape)
#     print(f1)
#搭建卷积神经网络注意力模块cbam
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=30):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv1d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.pad = nn.ZeroPad2d(padding=(0, 1, 0, 0))

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        avgout=self.pad(avgout)
        maxout=self.pad(maxout)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv1d(out))
        return out

#搭建卷积神经网络注意力模块cbam
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class exp_domcnn(nn.Module):
    def __init__(self):
        super(exp_domcnn, self).__init__()
        self.Conv1d=nn.Conv1d(in_channels=1,out_channels=6,kernel_size=5,stride=1)
        self.Conv1d1=nn.Conv1d(in_channels=6,out_channels=10,kernel_size=4,stride=2)
        self.relu=nn.ReLU()
        self.maxpool1d=nn.MaxPool1d(kernel_size=4,stride=1)
        self.lr = nn.LeakyReLU()
    def forward(self, x):
        out=self.Conv1d(x)
        out=self.lr(out)
        out=self.Conv1d1(out)
        out=self.lr(out)
        out=self.maxpool1d(out)
        return out



class sub_ortCNN(nn.Module):
    def __init__(self):
        super(sub_ortCNN,self).__init__()
        self.conv1=nn.Conv1d(in_channels=1,out_channels=20,kernel_size=1,stride=1)
        self.relu=nn.ReLU()
        self.lr = nn.LeakyReLU()
    def forward(self,x):
        out=self.conv1(x)
        out=self.lr(out)
        return out

class Deep_CNN(nn.Module):
    def __init__(self):
        super(Deep_CNN, self).__init__()
        self.conv1=nn.Conv1d(in_channels=60,out_channels=30,kernel_size=2,stride=2)
        self.relu = nn.ReLU()
        self.cbam=CBAM(channel=30)
        self.maxpool=nn.MaxPool1d(2,1)
        self.exp_dom=exp_domcnn()
        self.exp_dom1=exp_domcnn()
        self.sub_ort=sub_ortCNN()
        self.sub_ort1=sub_ortCNN()
        self.Linear1=nn.Linear(150,75)
        self.Linear2=nn.Linear(75,35)
        self.Linear3=nn.Linear(35,2)
        self.sigmode = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)
        self.lr=nn.LeakyReLU()
        self.bn=nn.BatchNorm1d(60)
    def forward(self,f1,f2,f3,f4):
        new_exp=self.exp_dom(f1)
        new_dom=self.exp_dom1(f2)
        new_ort=self.sub_ort(f3)
        new_sub=self.sub_ort1(f4)
        output=torch.cat([new_exp,new_dom,new_ort,new_sub], dim=1)
        output = self.bn(output)
        output=self.conv1(output)

        output=self.lr(output)
        output=self.cbam(output)
        output = self.lr(output)
        output=self.maxpool(output)
        output = output.view(-1, 150)
        output=self.Linear1(output)
        output = self.dropout(output)
        output=self.lr(output)
        output=self.Linear2(output)
        output=self.lr(output)
        output=self.Linear3(output)
        return output

my_model=Deep_CNN()
criterion = nn.CrossEntropyLoss()
path_checkpoint = "./improve cnn model_parameter/checkpoint_17_epochicc21.pkl"
checkpoint = torch.load(path_checkpoint, map_location='cpu')
my_model.load_state_dict(checkpoint['model_state_dict'])
my_model.eval()
scores1=[]
scores2=[]
test_loss=0
accuracy=0
total=0
TP=0
FP=0
TN=0
FN=0
tlabel=[]
test_losses= []
train_losses,test_losses= [], []
running_loss=0
# test_score=np.zeros((epochs*2,1019))
# test_label=np.zeros((epochs,1019))
with torch.no_grad():
    my_model.eval()
    for data2 in test_loader:
        fea1, fea2, fea3, fea4, labels = data2
        outputs = my_model(fea1, fea2, fea3, fea4)
        test_loss += criterion(outputs, labels.long())
        outputs = F.softmax(outputs, dim=1)
        print(outputs)
        _, predicted = torch.max(outputs.data, 1)


        total += labels.long().size(0)
    # print( labels.size(0))
        accuracy += (predicted == labels).sum()  # 标签预测对的个数
        TP += ((predicted == 1) & (labels == 1)).sum()
        TN += ((predicted == 0) & (labels == 0)).sum()
        FP += ((predicted == 1) & (labels == 0)).sum()
        FN += ((predicted == 0) & (labels == 1)).sum()
        routputs = np.array(outputs)
        outputs_col0 = routputs[:, 0]
        outputs_col1 = routputs[:, 1]
        outputs1 = outputs_col0.tolist()
        outputs2 = outputs_col1.tolist()
        label1 = np.array(labels)
        labels1 = label1.tolist()
        scores1 += outputs1
        scores2 += outputs2
        tlabel += labels1

# my_model.train()
train_losses.append(running_loss / len(train_loader))
test_losses.append(test_loss / len(test_loader))
sscores1 = np.array(scores1)
sscores2 = np.array(scores2)
print(sscores2)
tlabel1 = np.array(tlabel)
fpr, tpr, threshold = roc_curve(tlabel1, sscores2)
precision, recall, thresholds=precision_recall_curve(tlabel1,sscores2)
AP = average_precision_score(tlabel1,sscores2, average='macro', pos_label=1, sample_weight=None)
roc_auc = auc(fpr, tpr)  ###计算auc的值

# test_score[2 * e] = sscores1
# test_score[2 * e + 1] = sscores2
# test_label[e] = np.array(tlabel)
print(


      "测试准确率: {:.3f}.. ".format(accuracy / total),
      "精确率: {:.3f}.. ".format(TP / (TP + FP)),
      "召回率: {:.3f}.. ".format(TP / (TP + FN)),
      "F值: {:.3f}.. ".format((2 * TP) / (total + TP - TN)),
      "auc值: {:.3f}.. ".format(roc_auc))

#决策树模型
#数据处理
print("决策树模型")
Testsample = pd.read_excel(r'E:\testsample.xls')
Trainsample = pd.read_excel(r'E:\trainsample.xls')
Samples_all=Trainsample.append(Testsample)
rsamples=Samples_all.values
features_all=rsamples[:,0:96]
labels_all=rsamples[:,96]
train_feature,test_feature,train_label,test_label=train_test_split(features_all,labels_all,
test_size=0.2,random_state=1,stratify=labels_all)
clf = tree.DecisionTreeClassifier(criterion='entropy',max_features='log2',max_depth=9)# 载入决策树分类模型
model = clf.fit(train_feature, train_label)# 决策树拟合，得到模型
score = model.score(test_feature, test_label) #返回预测的准确度
# joblib.dump(model,'decisiontree.pkl')
print("Accuracy:", score)
#计算测试集的训练标签和auc值
predict_testlabel=model.predict(test_feature)
test_predictprobablity=model.predict_proba(test_feature)
fpr1,tpr1,threshold =roc_curve(test_label,test_predictprobablity[:,1])
roc_auc = auc(fpr1,tpr1)
print("auc值:", roc_auc)
precision1, recall1, thresholds=precision_recall_curve(test_label,test_predictprobablity[:,1])
AP_DT= average_precision_score(test_label,test_predictprobablity[:,1], average='macro', pos_label=1, sample_weight=None)
#计算混淆矩阵
model_confusion=metrics.confusion_matrix(test_label,predict_testlabel)
print("混淆矩阵:", model_confusion)
confusion_matrix=model_confusion
TN=confusion_matrix[0,0]
TP=confusion_matrix[1,1]
FN=confusion_matrix[1,0]
FP=confusion_matrix[0,1]
accuracy=(TP+TN)/(TP+TN+FN+FP)
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F_measure=(2*precision*recall)/(precision+recall)
print("accuracy:{}".format(accuracy),
      "precision:{}".format(precision),
      "recall:{}".format(recall),
      "F_measure:{} ".format(F_measure))


#adoboost模型
print("adoboost")
bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='entropy',max_features='log2',max_depth=8))
model = bdt.fit(train_feature, train_label)# 决策树拟合，得到模型
score = model.score(test_feature, test_label) #返回预测的准确度
print("Accuracy:", score)
#计算测试集的训练标签和auc值
predict_testlabel=model.predict(test_feature)
test_predictprobablity=model.predict_proba(test_feature)
fpr2,tpr2,threshold =roc_curve(test_label,test_predictprobablity[:,1])
roc_auc = auc(fpr2,tpr2)
print("auc值:", roc_auc)
precision2, recall2, thresholds=precision_recall_curve(test_label,test_predictprobablity[:,1])
AP_ado = average_precision_score(test_label,test_predictprobablity[:,1], average='macro', pos_label=1, sample_weight=None)
#计算混淆矩阵
model_confusion=metrics.confusion_matrix(test_label,predict_testlabel)
# print(test_label)
# print(predict_testlabel)
# print(type(test_label))
# print(type(predict_testlabel))
# print(test_label.shape)
# print(predict_testlabel.shape)
print("混淆矩阵:", model_confusion)
confusion_matrix=model_confusion
TN=confusion_matrix[0,0]
TP=confusion_matrix[1,1]
FN=confusion_matrix[1,0]
FP=confusion_matrix[0,1]
accuracy=(TP+TN)/(TP+TN+FN+FP)
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F_measure=(2*precision*recall)/(precision+recall)
print("accuracy:{}".format(accuracy),
      "precision:{}".format(precision),
      "recall:{}".format(recall),
      "F_measure:{} ".format(F_measure))

#svm模型
#训练svm分类器
print("支持向量机")
svc_clf = SVC( probability=True)
svc_clf=svc_clf.fit(train_feature, train_label)
# joblib.dump(svc_clf,'svm.pkl')
print("训练集：",svc_clf.score(train_feature,train_label))
print("测试集：",svc_clf.score(test_feature,test_label))
test_predict_label = svc_clf.decision_function(test_feature)
test_predictprobablity=svc_clf.predict_proba(test_feature)
fpr3,tpr3,threshold =roc_curve(test_label,test_predictprobablity[:,1])
roc_auc = auc(fpr3,tpr3)
print("auc值：",roc_auc)
precision3, recall3, thresholds=precision_recall_curve(test_label,test_predictprobablity[:,1])
AP_svm = average_precision_score(test_label,test_predictprobablity[:,1], average='macro', pos_label=1, sample_weight=None)
predictions = svc_clf.predict(test_feature)
confusion_matrix = metrics.confusion_matrix(test_label,predictions)
print(confusion_matrix)
# print(metrics.plot_confusion_matrix(svc_clf, test_feature, test_label))
TN=confusion_matrix[0,0]
TP=confusion_matrix[1,1]
FN=confusion_matrix[1,0]
FP=confusion_matrix[0,1]
accuracy=(TP+TN)/(TP+TN+FN+FP)
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F_measure=(2*precision*recall)/(precision+recall)
print("accuracy:{}".format(accuracy),
      "precision:{}".format(precision),
      "recall:{}".format(recall),
      "F_measure:{} ".format(F_measure))

#随机森林
print("随机森林")
clf = RandomForestClassifier(n_estimators=10)# 载入决策树分类模型
model = clf.fit(train_feature, train_label)# 决策树拟合，得到模型
score = model.score(test_feature, test_label) #返回预测的准确度
print("Accuracy:", score)
#计算测试集的训练标签和auc值
predict_testlabel=model.predict(test_feature)
test_predictprobablity=model.predict_proba(test_feature)
print(predict_testlabel)
fpr4,tpr4,threshold =roc_curve(test_label,test_predictprobablity[:,1])
roc_auc = auc(fpr4,tpr4)
print("auc值:", roc_auc)
precision4, recall4, thresholds=precision_recall_curve(test_label,test_predictprobablity[:,1])
AP_rt = average_precision_score(test_label,test_predictprobablity[:,1], average='macro', pos_label=1, sample_weight=None)
#计算混淆矩阵
model_confusion=metrics.confusion_matrix(test_label,predict_testlabel)
print("混淆矩阵:", model_confusion)
confusion_matrix=model_confusion
TN=confusion_matrix[0,0]
TP=confusion_matrix[1,1]
FN=confusion_matrix[1,0]
FP=confusion_matrix[0,1]
accuracy=(TP+TN)/(TP+TN+FN+FP)
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F_measure=(2*precision*recall)/(precision+recall)
print("accuracy:{}".format(accuracy),
      "precision:{}".format(precision),
      "recall:{}".format(recall),
      "F_measure:{} ".format(F_measure))
# plt.plot(fpr,tpr)

#贝叶斯模型
print("贝叶斯模型")
train_model=GaussianNB()
train_model.fit(train_feature,train_label)
print("Accuracy:", train_model.score(test_feature,test_label))
#计算训练集的预测标签
predict_testlabel=train_model.predict(test_feature)
test_predictprobablity=train_model.predict_proba(test_feature)
print(test_predictprobablity)
fpr5,tpr5,threshold =roc_curve(test_label,test_predictprobablity[:,1])
roc_auc = auc(fpr5,tpr5)
print("auc值:", roc_auc)
precision5, recall5, thresholds=precision_recall_curve(test_label,test_predictprobablity[:,1])
AP_BN = average_precision_score(test_label,test_predictprobablity[:,1], average='macro', pos_label=1, sample_weight=None)
#计算模型的混淆矩阵
model_confusion=metrics.confusion_matrix(test_label,predict_testlabel)
print("混淆矩阵:", model_confusion)
confusion_matrix=model_confusion
TN=confusion_matrix[0,0]
TP=confusion_matrix[1,1]
FN=confusion_matrix[1,0]
FP=confusion_matrix[0,1]
accuracy=(TP+TN)/(TP+TN+FN+FP)
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F_measure=(2*precision*recall)/(precision+recall)
print("accuracy:{}".format(accuracy),
      "precision:{}".format(precision),
      "recall:{}".format(recall),
      "F_measure:{} ".format(F_measure))


#Mlp方法
#导入数据集samples_data[:,0:96]
sample1 = pd.read_excel(r'E:\testsample.xls')
sample2 = pd.read_excel(r'E:\trainsample.xls')
samples_all=sample1.append(sample2)
# print(samples_all)
#划分数据集为测试集,训练集,验证集
samples_data=samples_all.values
# print(samples_data.shape)
X_data=samples_data[:,0:96]
Y_labels=samples_data[:,96]
X_other,X_test,Y_other,Y_test=train_test_split(X_data,Y_labels,test_size=0.2,random_state=1,stratify=Y_labels)
#reshape 标签特征
Y_test=Y_test.reshape((Y_test.shape[0],-1))
test_data=np.hstack((X_test,Y_test))
Y_other=Y_other.reshape((Y_other.shape[0],-1))
train_data=np.hstack((X_other,Y_other))
class DatasetFromNumPy(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)  # 返回长度

    def __getitem__(self, idx):
        # 这里可以对数据进行处理,比如讲字符数值化
        data = self.data[idx]  # 索引到第idx行的数据
        features1 = data[0:96]  # 前四项为特征数据
        # features2 = data[36:72]
        # features3 = data[72:84]
        # features4 = data[84:96]
        # features=features.reshape(1,8,12)
        features1 = torch.from_numpy(features1)
        # features2 = torch.from_numpy(features2)
        # features3 = torch.from_numpy(features3)
        # features4 = torch.from_numpy(features4)
        # features1= torch.unsqueeze(features1, dim=0)
        # features2=torch.unsqueeze(features2,dim=0)
        # features3=torch.unsqueeze(features3,dim=0)
        # features4=torch.unsqueeze(features4,dim=0)
        features1 = features1.to(torch.float32)
        # features2 = features2.to(torch.float32)
        # features3 = features3.to(torch.float32)
        # features4 = features4.to(torch.float32)
        # pad = nn.ZeroPad2d(padding=(0, 0, 0, 1))
        # features=pad(features1)
        label = data[96]  # 最后一项为指标数据
        label = torch.from_numpy(np.asarray(label))
        return features1, label  # 返回特征和指标
# DatasetFromNumPy 实例化，生成一个 trainLoader和test_loader，以便于数据处理
train_dataset = DatasetFromNumPy(train_data)
print(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=False)  # batch_size为每次读取的样本数量，shuffle是否选择随机读取数据
test_dataset=DatasetFromNumPy(test_data)
test_loader=DataLoader(test_dataset,batch_size=20,shuffle=True,drop_last=False)
my_model=mlp()
criterion = nn.CrossEntropyLoss()
path_checkpoint = "./mlpBestparameters/checkpoint_58_epoch.pkl"
checkpoint = torch.load(path_checkpoint, map_location='cpu')
my_model.load_state_dict(checkpoint['model_state_dict'])
my_model.eval()
scores1=[]
scores2=[]
test_loss=0
accuracy=0
total=0
TP=0
FP=0
TN=0
FN=0
tlabel=[]
test_losses= []
train_losses,test_losses= [], []
running_loss=0
with torch.no_grad():
    for data2 in test_loader:
        fea1, labels = data2
        outputs = my_model(fea1)
        test_loss += criterion(outputs, labels.long())
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.long().size(0)
        # print( labels.size(0))
        accuracy += (predicted == labels).sum()  # 标签预测对的个数
        TP += ((predicted == 1) & (labels == 1)).sum()
        TN += ((predicted == 0) & (labels == 0)).sum()
        FP += ((predicted == 1) & (labels == 0)).sum()
        FN += ((predicted == 0) & (labels == 1)).sum()
        routputs = np.array(outputs)
        outputs_col0 = routputs[:, 0]
        outputs_col1 = routputs[:, 1]
        outputs1 = outputs_col0.tolist()
        outputs2 = outputs_col1.tolist()
        label1 = np.array(labels)
        labels1 = label1.tolist()
        scores1 += outputs1
        scores2 += outputs2
        tlabel += labels1

# model.train()
train_losses.append(running_loss / len(train_loader))
test_losses.append(test_loss / len(test_loader))
sscores1 = np.array(scores1)
sscores2 = np.array(scores2)
tlabel1 = np.array(tlabel)
fpr10, tpr10, threshold = roc_curve(tlabel1, sscores2)
roc_auc10 = auc(fpr10, tpr10)  ###计算auc的值
precision9, recall9, thresholds = precision_recall_curve(tlabel1, sscores2)
AP10 = average_precision_score(tlabel1, sscores2, average='macro', pos_label=1, sample_weight=None)
# test_score[2 * e] = sscores1
# test_score[2 * e + 1] = sscores2
# test_label[e] = np.array(tlabel)
print(
      "测试准确率: {:.3f}.. ".format(accuracy / total),
      "精确率: {:.3f}.. ".format(TP / (TP + FP)),
      "召回率: {:.3f}.. ".format(TP / (TP + FN)),
      "F值: {:.3f}.. ".format((2 * TP) / (total + TP - TN)),
      "auc值: {:.3f}.. ".format(roc_auc10),
      "AP值:{:.3f}..".format(AP10))

#DeepEP模型
protein_feature=pd.read_excel(r'./data/pro_attri.xls')

noess_sample=protein_feature[protein_feature['label']==0]
noess_sample=noess_sample.values
X_noess=noess_sample[:,0:100]
Y_noess=noess_sample[:,100]
# print(noess_sample)
ess_sample=protein_feature[protein_feature['label']==1]
ess_sample=ess_sample.values
X_ess=ess_sample[:,0:100]
Y_ess=ess_sample[:,100]
#非关键蛋白的划分
X_trainnoess,X_testnoess,Y_trainnoess,Y_testnoess=train_test_split(X_noess,Y_noess,test_size=0.2,random_state=1)
#关键蛋白的划分
X_trainess,X_testess,Y_trainess,Y_testess=train_test_split(X_ess,Y_ess,test_size=0.2,random_state=1)
#测试集的构建
Y_testnoess=Y_testnoess.reshape((Y_testnoess.shape[0],-1))
testnoess=np.hstack((X_testnoess,Y_testnoess))
Y_testess=Y_testess.reshape((Y_testess.shape[0],-1))
print(Y_testess.sum())
testess=np.hstack((X_testess,Y_testess))
test_data=np.vstack((testnoess,testess))
print(test_data.shape)
#训练集的构建
Y_trainnoess=Y_trainnoess.reshape((Y_trainnoess.shape[0],-1))
trainnoess=np.hstack((X_trainnoess,Y_trainnoess))
Y_trainess=Y_trainess.reshape((Y_trainess.shape[0],-1))
trainess=np.hstack((X_trainess,Y_trainess))
print(Y_trainess.sum())
# row_rand_array = np.arange(2355)
# print(row_rand_array)
# np.random.shuffle(row_rand_array)#将3行对应的3个index [0 1 2] 打乱
# print(row_rand_array)
# noess_random= trainnoess[row_rand_array[0:1167]]
# train_data=np.vstack((trainnoess,trainess))
# print(train_data.shape)
class DatasetFromNumPy(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)  # 返回长度

    def __getitem__(self, idx):
        # 这里可以对数据进行处理,比如讲字符数值化
        data = self.data[idx]  # 索引到第idx行的数据
        features1 = data[0:36]  # 前四项为特征数据
        features2 = data[36:100]
        features1=features1.reshape(3,12)
        features1 = torch.from_numpy(features1)
        rfeatures1=features1.transpose(0,1)
        features1=rfeatures1.reshape(3,12)
        # features1 = torch.from_numpy(features1)
        features2 = torch.from_numpy(features2)
        features2=features2.to(torch.float32)
        features1 = features1.to(torch.float32)

        # features1= torch.squeeze(features1, dim=2)
        #features2=torch.unsqueeze(features2,dim=0)
        # pad = nn.ZeroPad2d(padding=(0, 0, 0, 1))
        # features=pad(features1)
        label = data[100]  # 最后一项为指标数据
        label = torch.from_numpy(np.asarray(label))
        return features1, features2, label  # 返回特征和指标
# train_dataset = DatasetFromNumPy(train_data)
# train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, drop_last=False)
test_dataset=DatasetFromNumPy(test_data)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True,drop_last=False)
my_model=Deepep()
criterion = nn.CrossEntropyLoss()
path_checkpoint = "./deepep_good_parameter/checkpoint_36_epoch.pkl"
checkpoint = torch.load(path_checkpoint, map_location='cpu')
my_model.load_state_dict(checkpoint['model_state_dict'])
my_model.eval()
scores1=[]
scores2=[]
test_loss=0
accuracy=0
total=0
TP=0
FP=0
TN=0
FN=0
tlabel=[]
test_losses= []
train_losses,test_losses= [], []
running_loss=0
# test_score=np.zeros((epochs*2,1019))
# test_label=np.zeros((epochs,1019))
with torch.no_grad():
    my_model.eval()
    for data2 in test_loader:
        fea1, fea2,  labels = data2
        outputs = my_model(fea1, fea2)
        test_loss += criterion(outputs, labels.long())
        outputs = F.softmax(outputs, dim=1)
        # print(outputs)
        _, predicted = torch.max(outputs.data, 1)


        total += labels.long().size(0)
    # print( labels.size(0))
        accuracy += (predicted == labels).sum()  # 标签预测对的个数
        TP += ((predicted == 1) & (labels == 1)).sum()
        TN += ((predicted == 0) & (labels == 0)).sum()
        FP += ((predicted == 1) & (labels == 0)).sum()
        FN += ((predicted == 0) & (labels == 1)).sum()
        routputs = np.array(outputs)
        outputs_col0 = routputs[:, 0]
        outputs_col1 = routputs[:, 1]
        outputs1 = outputs_col0.tolist()
        outputs2 = outputs_col1.tolist()
        label1 = np.array(labels)
        labels1 = label1.tolist()
        scores1 += outputs1
        scores2 += outputs2
        tlabel += labels1

# my_model.train()
# train_losses.append(running_loss / len(train_loader))
test_losses.append(test_loss / len(test_loader))
sscores1 = np.array(scores1)
sscores2 = np.array(scores2)
print(sscores2)
tlabel1 = np.array(tlabel)
fpr11, tpr11, threshold = roc_curve(tlabel1, sscores2)
precision11, recall11, thresholds=precision_recall_curve(tlabel1,sscores2)
AP11 = average_precision_score(tlabel1,sscores2, average='macro', pos_label=1, sample_weight=None)
roc_auc11 = auc(fpr11, tpr11)  ###计算auc的值

# test_score[2 * e] = sscores1
# test_score[2 * e + 1] = sscores2
# test_label[e] = np.array(tlabel)
print(


      "测试准确率: {:.3f}.. ".format(accuracy / total),
      "精确率: {:.3f}.. ".format(TP / (TP + FP)),
      "召回率: {:.3f}.. ".format(TP / (TP + FN)),
      "F值: {:.3f}.. ".format((2 * TP) / (total + TP - TN)),
      "auc值: {:.3f}.. ".format(roc_auc11),
        "AP值：{:.3f}..".format(AP11))

plt.plot(fpr,tpr,label='DLAM(AUC = %0.3f)' % auc(fpr,tpr))
# plt.plot(fpr1,tpr1,label='Decision tree(AUC = %0.3f)' % auc(fpr1,tpr1))
plt.plot(fpr3,tpr3,label='SVM(AUC = %0.3f)' % auc(fpr3,tpr3))
plt.plot(fpr5,tpr5,label='Naïve Bayes(AUC = %0.3f)' % auc(fpr5,tpr5))
plt.plot(fpr2,tpr2,label='Adaboost(AUC = %0.3f)' % auc(fpr2,tpr2))
plt.plot(fpr10,tpr10,label='MLP(AUC = %0.3f)'% auc(fpr10,tpr10))
plt.plot(fpr11,tpr11,label='DeepEP(AUC = %0.3f)'% auc(fpr11,tpr11))
# plt.plot(fpr4,tpr4,label='Random forest (AUC = %0.3f)' % auc(fpr4,tpr4))

plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
#
precision, recall, thresholds = precision_recall_curve(tlabel1,sscores2)
plt.plot(recall,precision,label='DLAM(AP = %0.3f)' %AP)
# plt.plot(recall1,precision1,label='Decision tree(AP = %0.3f)' %AP_DT)
plt.plot(recall3,precision3,label='SVM(AP = %0.3f)' %AP_svm)
plt.plot(recall5,precision5,label='Naïve Bayes(AP = %0.3f)' %AP_BN)
plt.plot(recall2,precision2,label='Adaboost(AP = %0.3f)' %AP_ado)
plt.plot(recall9,precision9,label='MLP(AP = %0.3f)' %AP10)
plt.plot(recall11,precision11,label='DeepEP(AP = %0.3f)' %AP11)
# plt.plot(recall4,precision4,label='Random forest(AP = %0.3f)' %AP_rt)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="upper right")
plt.show()

central_scores=pd.read_excel(r'data/scores.xls');
Wpdinm=pd.read_excel(r'data/DIPscore.xls');
WPDINM=Wpdinm['wpdinm']
WPDINM=WPDINM.values
LABLE=Wpdinm['isessential']
LABLE=LABLE.values
# print(central_scores)
DC=central_scores['DC']
DC=DC.values
IC=central_scores['IC']
IC=IC.values
SC=central_scores['SC']
SC=SC.values
CC=central_scores['CC']
CC=CC.values
NC=central_scores['NC']
NC=NC.values
CoEWC=central_scores['CoEWC']
CoEWC=CoEWC.values
ION=central_scores['ION']
ION=ION.values
TEGS=central_scores['TEGS']
TEGS=TEGS.values
essential=central_scores['essential']
essential=essential.values

# print(essential.shape[0])
# essential=essential.reshape((essential.shape[0],-1))
# print(essential.shape)
# DC=DC.reshape((DC.shape[0],-1))
fpr1,tpr1,threshold=roc_curve(essential,DC)
fpr2,tpr2,threshold=roc_curve(essential,IC)
fpr3,tpr3,threshold=roc_curve(essential,SC)
fpr4,tpr4,threshold=roc_curve(essential,CC)
fpr5,tpr5,threshold=roc_curve(essential,NC)
fpr6,tpr6,threshold=roc_curve(essential,CoEWC)
fpr7,tpr7,threshold=roc_curve(essential,ION)
fpr8,tpr8,threshold=roc_curve(essential,TEGS)
fpr9,tpr9,threshold=roc_curve(LABLE,WPDINM)
plt.plot(fpr,tpr,color='r',label='DLAM(AUC = %0.3f)' % auc(fpr,tpr))
plt.plot(fpr2,tpr2,label='IC(AUC = %0.3f)' % auc(fpr2,tpr2))
plt.plot(fpr4,tpr4,label='CC(AUC = %0.3f)' % auc(fpr4,tpr4))
plt.plot(fpr3,tpr3,label='SC(AUC = %0.3f)' % auc(fpr3,tpr3))
plt.plot(fpr1,tpr1,color='k',label='DC(AUC = %0.3f)' % auc(fpr1,tpr1))
plt.legend(loc="lower right")
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()

plt.plot(fpr5,tpr5,  label='NC(AUC = %0.3f)' % auc(fpr5,tpr5))
plt.plot(fpr6,tpr6,label='CoEWC(AUC = %0.3f)' % auc(fpr6,tpr6))
plt.plot(fpr7,tpr7,label='ION(AUC = %0.3f)' % auc(fpr7,tpr7))
plt.plot(fpr8,tpr8,color='k',label='TEGS(AUC = %0.3f)' % auc(fpr8,tpr8))
plt.plot(fpr,tpr,color='r',label='DLAM(AUC = %0.3f)' % auc(fpr,tpr))
# plt.plot(fpr9,tpr9,color='gold',label='WPDINM(AUC = %0.3f)' % auc(fpr9,tpr9))
plt.legend(loc="lower right")
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()

precision, recall, thresholds = precision_recall_curve(tlabel1,sscores2)
precision1, recall1, thresholds = precision_recall_curve(essential,DC)
precision2, recall2, thresholds = precision_recall_curve(essential,IC)
precision3, recall3, thresholds = precision_recall_curve(essential,SC)
precision4, recall4, thresholds = precision_recall_curve(essential,CC)
precision5, recall5, thresholds = precision_recall_curve(essential,NC)
precision6, recall6, thresholds = precision_recall_curve(essential,CoEWC)
precision7, recall7, thresholds = precision_recall_curve(essential,ION)
precision8, recall8, thresholds = precision_recall_curve(essential,TEGS)
precision9, recall9, thresholds = precision_recall_curve(LABLE,WPDINM)
AP1 = average_precision_score(essential,DC, average='macro', pos_label=1, sample_weight=None)
AP2 = average_precision_score(essential,IC, average='macro', pos_label=1, sample_weight=None)
AP3 = average_precision_score(essential,SC, average='macro', pos_label=1, sample_weight=None)
AP4 = average_precision_score(essential,CC, average='macro', pos_label=1, sample_weight=None)
AP5 = average_precision_score(essential,NC, average='macro', pos_label=1, sample_weight=None)
AP6 = average_precision_score(essential,CoEWC, average='macro', pos_label=1, sample_weight=None)
AP7 = average_precision_score(essential,ION, average='macro', pos_label=1, sample_weight=None)
AP8 = average_precision_score(essential,TEGS, average='macro', pos_label=1, sample_weight=None)

plt.plot(recall,precision,color='r',label='DLAM(AP = %0.3f)' % AP)
plt.plot(recall1,precision1,label='DC(AP = %0.3f)' %AP1)
plt.plot(recall2,precision2,label='IC(AP = %0.3f)' %AP2)
plt.plot(recall3,precision3,label='SC(AP = %0.3f)' %AP3)
plt.plot(recall4,precision4,color='k',label='CC(AP = %0.3f)' %AP4)

plt.legend(loc="upper right")
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.show()

plt.plot(recall,precision,color='r',label='DLAM(AP = %0.3f)' %AP)
plt.plot(recall5,precision5,label='NC(AP = %0.3f)' %AP5)
plt.plot(recall6,precision6,label='CoEWC(AP = %0.3f)' %AP6)
plt.plot(recall7,precision7,label='ION(AP = %0.3f)' %AP7)
plt.plot(recall8,precision8,color='k',label='TEGS(AP = %0.3f)' %AP8)
# plt.plot(recall9,precision9,color='gold',label='WPDINM(AP = %0.3f)' )
plt.legend(loc="upper right")
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.show()

