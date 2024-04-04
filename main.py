from Linear import *
from ActivationFunctions import *
from LossFunctions import *
from Optimizers import *
from Network import *
from DataLoader import *

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt


#Example neural network data preprocessing (predicting heart disease based on data)
txt = open(r'heart.dat')
txt_list = [line.replace('\n','').split(' ') for line in txt.readlines()]
df = pd.DataFrame(txt_list, columns=['age','sex','chestpaintype','resting blood pressure', 'serum cholestoral', 'fasting blood sugar', 'restingelectrocardiographicresults', 'maximum heart rate achieved', 'exercise induced angina', 'oldpeak', 'slope of the peak exercise', 'number of major vessels', 'thal', 'heartdisease'])
ohe = OneHotEncoder()
feature_arry = ohe.fit_transform(df[["chestpaintype","restingelectrocardiographicresults","thal"]]).toarray()
feature_labels=np.array(["chestpaintype_1","chestpaintype_2","chestpaintype_3","chestpaintype_4","restingelectrocardiographicresults_1","restingelectrocardiographicresults_2","restingelectrocardiographicresults_3","thal_1","thal_2","thal_3"])
features = pd.DataFrame(feature_arry, columns = feature_labels)
df_new = pd.concat([df[['age','sex','resting blood pressure', 'serum cholestoral', 'fasting blood sugar', 'maximum heart rate achieved', 'exercise induced angina', 'oldpeak', 'slope of the peak exercise', 'number of major vessels', 'heartdisease']], features], axis=1)
train=df_new.sample(frac=0.8,random_state=1)
test=df_new.drop(train.index)

train=train.astype(float)
test=test.astype(float)

heartdisease=train.heartdisease
train=train.drop('heartdisease',axis=1)
heartdisease_test=test.heartdisease
test=test.drop('heartdisease',axis=1)

test=test.astype(float)
train=train.astype(float)

minimum=train.min()
maximum=train.max()

train=((train-minimum)/(maximum-minimum))
test=((test-minimum)/(maximum-minimum))

train['heartdisease']=heartdisease
test['heartdisease']=heartdisease_test

test=test.astype(float)
train=train.astype(float)

train = train.to_numpy()
test = test.to_numpy()

train[:,-1]-=1
test[:,-1]-=1


#Neural network parameters
epochs = 200
batch_size_val=1

train_loader = DataLoader(inputs=train[:, :-1],labels=train[:,-1].reshape(-1, 1),batch_size=batch_size_val,shuffle=True,drop_last=False)
test_loader = DataLoader(inputs=test[:, :-1],labels=test[:,-1].reshape(-1, 1),batch_size=1,shuffle=False,drop_last=False)

net = Network(loss=MSELoss,optimizer=SGD())

net.add_layer(layer=Linear,layer_inputs=20,layer_outputs=12,distribution="normal")
net.add_activation(LeakyRelu(a=0.001))
net.add_layer(layer=Linear,layer_inputs=12,layer_outputs=12,distribution="normal")
net.add_activation(LeakyRelu(a=0.001))
net.add_layer(layer=Linear,layer_inputs=12,layer_outputs=1,distribution="normal")
net.add_activation(Sigmoid())

##########################
loss_array=[]
TP_array=[] # true positive
TN_array=[] # true negative
FP_array=[] # false positive
FN_array=[] # false negative

for epoch in range(epochs):
    #training
    net.train_mode()
    loss_array_epoch=[]
    for inputs,labels in train_loader:
        loss_array_epoch.append(net.train(inputs,labels))
    loss_array.append(np.array(loss_array_epoch).mean())

    #evaluation
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    net.evaluation_mode()
    for inputs,labels in test_loader:
        eval = np.round(net.evaluate(inputs))
        if (eval == labels):
            if(labels==1):
                TP+=1
            elif(labels==0):
                TN+=1
        else:
            if (labels == 1):
                FN += 1
            elif (labels == 0):
                FP += 1

    TP_array.append(TP)
    TN_array.append(TN)
    FP_array.append(FP)
    FN_array.append(FN)


#Various indicators for true/false neural network
# sensitivity=TP_array/(TP_array+FN_array)
# specificity=TN_array/(FP_array+TN_array)
# false_positive_rate=FP_array/(FP_array+TN_array)
# false_discovery_rate=FP_array/(FP_array+TP_array)
# positive_precision=TP_array/(TP_array+FP_array)
# negative_precision=TN_array/(TN_array+FN_array)
# accuracy=(TP_array+TN_array)/(TP_array+TN_array+FP_array+FN_array)
# f1_score=(2*TP_array)/(2*TP_array+FP_array+FN_array)
# MCC=(TP_array*TN_array-FP_array*FN_array)/(sqrt((TP_array+FP_array)*(TP_array+FN_array)*(TN_array+FP_array)*(TN_array+FN_array)))
# beta=5
# fb_score=(1+5)*(positive_precision*sensitivity)/((1+5)*positive_precision+sensitivity)

#Charts
length = [i for i in range(len(loss_array))]
plt.plot(length, loss_array)

#length = [i for i in range(len(TP_array))]
#plt.plot(length, TP_array, 'r', length, TN_array, 'b', length, FP_array, 'g',length, FN_array, 'k')
#plt.legend(["TP","TN","FP","FN"])
plt.show()

