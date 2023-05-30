from Linear import *
from ActivationFunctions import *
from LossFunctions import *
from Optimizers import *
from Network import *
from DataLoader import *


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from math import sqrt

#size - 216(treningowe)+54(testowe)=270

#preprocessing
txt = open(r'C:\Users\karol\Desktop\SNB\Statlog_(Heart)_MLR\heart.dat')
txt_list = [line.replace('\n','').split(' ') for line in txt.readlines()]
df = pd.DataFrame(txt_list, columns=['age','sex','chestpaintype','resting blood pressure', 'serum cholestoral', 'fasting blood sugar', 'restingelectrocardiographicresults', 'maximum heart rate achieved', 'exercise induced angina', 'oldpeak', 'slope of the peak exercise', 'number of major vessels', 'thal', 'heartdisease'])

#ONEHOT
ohe = OneHotEncoder()
feature_arry = ohe.fit_transform(df[["chestpaintype","restingelectrocardiographicresults","thal"]]).toarray()
feature_labels=np.array(["chestpaintype_1","chestpaintype_2","chestpaintype_3","chestpaintype_4","restingelectrocardiographicresults_1","restingelectrocardiographicresults_2","restingelectrocardiographicresults_3","thal_1","thal_2","thal_3"])

features = pd.DataFrame(feature_arry, columns = feature_labels)
df_new = pd.concat([df[['age','sex','resting blood pressure', 'serum cholestoral', 'fasting blood sugar', 'maximum heart rate achieved', 'exercise induced angina', 'oldpeak', 'slope of the peak exercise', 'number of major vessels', 'heartdisease']], features], axis=1)

train=df_new.sample(frac=0.8,random_state=1) #podzielenie danych na treningowe oraz testowe
test=df_new.drop(train.index)
test=test.astype(float)
train=train.astype(float)

heartdisease=train.heartdisease
train=train.drop('heartdisease',axis=1)
heartdisease_test=test.heartdisease
test=test.drop('heartdisease',axis=1)

test=test.astype(float)
train=train.astype(float)
minimum=train.min()
maximum=train.max()
train=((train-minimum)/(maximum-minimum)) #normalizacja

test=((test-minimum)/(maximum-minimum)) #normalizacja
train['heartdisease']=heartdisease

test['heartdisease']=heartdisease_test
test=test.astype(float)
train=train.astype(float)

train = train.to_numpy()
test = test.to_numpy()

train[:,-1]-=1
test[:,-1]-=1


#Ustalanie parametrów sieci neuoronowej 
epochs = 50
batch_size_val=1

train_loader = DataLoader(inputs=train[:, :-1],labels=train[:,-1].reshape(-1, 1),batch_size=batch_size_val,shuffle=True,drop_last=False)
test_loader = DataLoader(inputs=test[:, :-1],labels=test[:,-1].reshape(-1, 1),batch_size=1,shuffle=False,drop_last=False)

net = Network(loss=BCELoss,optimizer=SGD())
net.add_layer(layer=Linear,layer_inputs=20,layer_outputs=12,distribution="normal")
net.add_activation(LeakyRelu(a=0.001))
net.add_layer(layer=Linear,layer_inputs=12,layer_outputs=12,distribution="normal")
net.add_activation(LeakyRelu(a=0.001))
net.add_layer(layer=Linear,layer_inputs=12,layer_outputs=1,distribution="normal")
net.add_activation(Sigmoid())

##########################
loss_array=[]
TP_array=[] # true- positive
TN_array=[] # true- negative
FP_array=[] # false- positive
FN_array=[] # false- negative 

for epoch in range(epochs):
    #TRENING
    net.train_mode()
    loss_array_epoch=[]
    for inputs,labels in train_loader:
        loss_array_epoch.append(net.train(inputs,labels))
    loss_array.append(np.array(loss_array_epoch).mean())

    #EWALUACJA
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    net.evaluation_mode()
    for inputs,labels in test_loader:
        eval = np.round(net.evaluate(inputs))
        #print("EVAL",len(eval))
        #print("LABELS",labels)
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

    #zrobić arraye,appendować do nich i potem zrobić plot
    #te dwa mają być w sprawku
    sensitivity=TP/(TP+FN) #czułość
    specificity=TN/(FP+TN) #specyficzność
    print(sensitivity)
    print(specificity)

    #dodatkowe miary, ewentualnie można dodać
    false_positive_rate=FP/(FP+TN) #częstość fałszywych alarmów
    false_discovery_rate=FP/(FP+TP) #częstość fałszywych odkryć

    positive_precision=TP/(TP+FP) #precyzja pozytywna(jedynek-obecności choroby)
    negative_precision=TN/(TN+FN) #precyzja negatywna(zer-braku obecności choroby)

    accuracy=(TP+TN)/(TP+TN+FP+TN) #dokładność

    f1_score=(2*TP)/(2*TP+FP+FN) #średnia harmoniczna z precyzji i czułości

    MCC=(TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))) #współczynnik korelacji matthews (matthews correlation coefficient)

    #To się może przydać, bo mamy inną wagę fałszywych jedynek i zer w opisie zadania
    beta=5 #waga źle sklasyfikowanych jedynek ma być 5 razy większa niż źle sklasyfikowanych zer, jeśli dobrze pamiętam
    fb_score=(1+5)*(positive_precision*sensitivity)/((1+5)*positive_precision+sensitivity)




    TP_array.append(TP)
    TN_array.append(TN)
    FP_array.append(FP)
    FN_array.append(FN)


#WYKRESY STRAT, DOKŁADNOŚCI ITD.
#length = [i for i in range(len(TP_array))]
length = [i for i in range(len(loss_array))]
plt.plot(length, loss_array)
#plt.plot(length, TP_array, 'r', length, TN_array, 'b', length, FP_array, 'g',length, FN_array, 'k')
#plt.legend(["TP","TN","FP","FN"])
plt.show()

