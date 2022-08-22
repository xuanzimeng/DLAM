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

#导入数据集samples_data[:,0:96]
sample1 = pd.read_excel(r'.\data\testsample.xls')
sample2 = pd.read_excel(r'.\data\trainsample.xls')
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
        return out;



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

#训练
patience = 10	# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)
#对CNN_CNAM进行实例化
model=Deep_CNN()
#定义损失函数,交叉熵函数
criterion = nn.CrossEntropyLoss()
#设置优化方法
optimizer = torch.optim.Adam(model.parameters(),lr=0.009 )

#设置训练轮数
epochs=200
#训练误差和测试误差都加入到下面的列表里
train_losses,test_losses,vali_losses= [], [],[]
test_score=np.zeros((epochs*2,1019))
test_label=np.zeros((epochs,1019))

#开始训练
print('开始训练')
for e in range(epochs):

    running_loss=0
    total1=0
    scores1=[]
    scores2=[]
    tlabel=[]
    #将迭代器中小批量的特征图导入进来

    for data1 in train_loader:
        model.train()
        #优化器中所有的导数设为零
        optimizer.zero_grad()
        fe1,fe2,fe3,fe4,labels1=data1
        #对这小批量的特征矩阵进行正向传播
        outputs=model(fe1,fe2,fe3,fe4)
        #计算出预测的误差
        loss=criterion(outputs,labels1.long())
        total1 += labels1.long().size(0)
        #然后进行反向传播
        loss.backward()
        #进行优化迭代,更新权重
        optimizer.step()
        #最后将损失加进来
        running_loss+=loss.item()

        checkpoint = {"model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),"epoch": e}
        path_checkpoint = "./DLAM_parameter/checkpoint_{}_epoch.pkl".format(e)
        torch.save(checkpoint, path_checkpoint)




    else:
       test_loss=0
       accuracy=0
       total=0
       TP=0
       FP=0
       TN=0
       FN=0
        #测试的时候不需要反向传播
       with torch.no_grad():
         model.eval()
         feat1,feat2,feat3,feat4, labelss = next(iter(validate_loader))
         valid_output = model(feat1,feat2,feat3,feat4)
         valid_loss = criterion(valid_output, labelss.long())  # 注意这里的输入参数维度要符合要求，我这里为了简单，并未考虑这一点
         vali_losses.append(valid_loss)
         early_stopping(valid_loss, model)
         # 若满足 early stopping 要求
         if early_stopping.early_stop:
             print("Early stopping")
             # 结束模型训练
             break


         for data2 in test_loader:
             fea1,fea2,fea3,fea4,labels=data2
             outputs=model(fea1,fea2,fea3,fea4)
             test_loss+=criterion(outputs,labels.long())
             _, predicted = torch.max(outputs.data, dim=1)
             total += labels.long().size(0)
             # print( labels.size(0))
             accuracy += (predicted == labels).sum()  #标签预测对的个数
             TP+=((predicted==1)&(labels==1)).sum()
             TN+=((predicted==0)&(labels==0)).sum()
             FP+=((predicted==1)&(labels==0)).sum()
             FN+=((predicted==0)&(labels==1)).sum()
             routputs=np.array(outputs)
             outputs_col0=routputs[:,0]
             outputs_col1=routputs[:,1]
             outputs1=outputs_col0.tolist()
             outputs2=outputs_col1.tolist()
             label1=np.array(labels)
             labels1=label1.tolist()
             scores1+=outputs1
             scores2+=outputs2
             tlabel+=labels1

    # model.train()
    train_losses.append(running_loss/len(train_loader))
    test_losses.append(test_loss/len(test_loader))
    sscores1=np.array(scores1)
    sscores2=np.array(scores2)
    tlabel1=np.array(tlabel)
    fpr, tpr, threshold=roc_curve(tlabel1,sscores2)
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    test_score[2*e]=sscores1
    test_score[2*e+1]=sscores2
    test_label[e]=np.array(tlabel)
    print("训练集学习次数: {}/{}..".format(e+1,epochs),
                "训练误差: {:.3f}.. ".format(running_loss/len(train_loader)),
                 "测试误差: {:.3f}.. ".format(test_loss/len(test_loader)),
                  "测试准确率: {:.3f}.. ".format(accuracy/total),
      "精确率: {:.3f}.. ".format(TP/(TP+FP)),
       "召回率: {:.3f}.. ".format(TP/(TP+FN)),
                "F值: {:.3f}.. ".format((2*TP)/(total+TP-TN)),
          "auc值: {:.3f}.. ".format(roc_auc))


plt.plot(train_losses,label='Training loss',marker='o')
plt.plot(test_losses,label='Test loss',marker='o')
plt.plot(vali_losses,label='Validation loss',marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(axis='both')
plt.legend()
plt.show()









