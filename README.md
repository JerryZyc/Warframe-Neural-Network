# DNN Neural Network for 'Warframe' Price Prediction

by Yancheng Zhu(yz3365) & Shili Wu(sw3302)

# Problem Description
We are now playing a game called 'warframe', and there is a trading webside for this game named 'warframe markect'(https://warframe.market/).

Suppose that I'm a businessman who want to know how products' price would change in the future, it would be very 
helpful for making trading strategy if I have a good model to predict the price.

Artificial neural network is data-driven and self-adaptive in nature to solve the problem existing in the classical
time series prediction. The desired model is adaptively formed based on the features presented from the data. 
The most widely used one in the forecasting problems are the multi-layer perceptrons(MLPs), which the network
is consist of at least three layers- input, one or more hidden and output layer. 

Based on the trading data collected from the website, we would like to build a DNN model to predict products' price
in 3 month. 

Here is the python code.

# Part1 Data Collection
## Step1 Data Scrapy

Open the Jupyter notebook and import scrapy modules like 'requests', 'html' and 'bs4'.
```pythonscript
from bs4 import BeautifulSoup
from lxml import html
from lxml import etree
import xml
import requests
```

Next, open the webside for trading a 'Nova Prime'(https://warframe.market/items/nova_prime_set) and 
view the source. Copy the url of the website to creat a request.get.

```pythonscript
url = "https://warframe.market/items/nova_prime_set"
r = requests.get(url).text
s = etree.HTML(r)
trade = s[4].text
```
Then, the trading information is download as a text file.

## Step2 Data Transform

The text is a json file. We should transform it to a dictionary to read the content. Let's import mudole to deal with it.

```pythonscript   
import json
import pandas as pd
from pandas.io.json import json_normalize
```
```pythonscript
data=json.loads(trade)
data1=data['payload']['statistics_closed']['90days']
```
So, here data1 is a dictionary I need as trading information. The details of its element can be shown as follows:

[{'datetime': '2019-02-09T00:00:00.000+00:00', 'volume': 95, 'min_price': 100, 'max_price': 120, 'open_price':
110, 'closed_price': 120, 'avg_price': 110.0, 'wa_price': 109.379, 'median': 110, 'moving_avg': 114.6, 'donch_top':
130, 'donch_bot': 100, 'id': '5c5f6a863ae3e80024bca024'}, {'datetime': '2019-02-10T00:00:00.000+00:00', 'volume':
62, 'min_price': 100, 'max_price': 120, 'open_price': 110, 'closed_price': 120, 'avg_price': 110.0, 'wa_price':
10


The next step is to read it and generate a table to store the things I am interested in.

Import data science modules.
```pythonscript
import matplotlib
matplotlib.use('Agg')
from datascience import Table
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight')
```
Then, choose the information I need. Assume I want to know the max_price, avg_prive, trading volume and so on, 
creat several arrays to store them.
```pythonscript
time=np.array([])
volume=np.array([])
min_price=np.array([])
max_price=np.array([])
avg_price=np.array([])
median_price=np.array([])
warframe=np.array([])
```
Read the dictionary to input elements. Here I plan to get 90 orders for one kind of product called 'Nova'.
```pythonscript
for i in range (0,90):
    time=np.append(time,data1[i]['datetime'][0:10])
    volume=np.append(volume,data1[i]['volume'])
    min_price=np.append(min_price,data1[i]['min_price'])
    max_price=np.append(max_price,data1[i]['max_price'])
    avg_price=np.append(avg_price,data1[i]['avg_price'])
    median_price=np.append(median_price,data1[i]['median'])
    warframe=np.append(warframe,'nova')
```
Finally, we generate the table with these arrays.
```pythonscript
trading=Table().with_columns('time',time,'volume',volume,'min_price',min_price,'max_price',max_price,'avg_price',avg_price,'median_price',median_price,'warframe',warframe)
```
Store the talbe as a csv file.
```pythonscript
trading.to_csv('Nova Prime.csv')
```

## Step3 Data Organization

After collecting 15 kinds of products to build training data set, we need to combine these 15 csv files as one. 
Here is a module named 'glob' that can help us. 
```pythonscript
import glob
```
Then, we  got a trainging dataset named 'training_data.csv', with 1336 rows of trading information.
```pythonscript
csvx_list = glob.glob('*.csv')
for i in csvx_list:
    fr = open(i,'r').read()
    with open('training_data.csv','a') as f:
        f.write(fr)
```

# Part2 Nerual Network Model
## Step1 Build Training Dateset
Import the training dataset 'warframe_data.csv' as a 'dataframe' structure, and then transform it to a 'table' structure. 
```pythonscript
training = pd.read_csv('warframe_data.csv')
Train=Table.from_df(training)
Train
```
I have to mention that the we add many other features based on the market and players' daily comments,
like the easiness to get the item, the price of the raw material needed, the level of demanding and so on. 
These can help us training the DNN model better. So now we have 7 features as the input of DNN.

```pythonscript
features_training=[]
for i in range(1335):
    feature1=[]
    feature1.append(Train.row(i)[7])
    feature1.append(Train.row(i)[8]/100)
    feature1.append(Train.row(i)[9])
    feature1.append(Train.row(i)[10])
    feature1.append(Train.row(i)[11]/100)
    feature1.append(Train.row(i)[12])
    feature1.append(Train.row(i)[13])
    features_training.append(feature1)
```
And we choose to predict the ave_price and daily trading volume of the product. So, there are 2 labels as the output.

```pythonscript
labels_training=[]
for i in range(1335):
    label=[]
    label.append(Train.row(i)[1])
    label.append(Train.row(i)[4])
    labels_training.append(label)
```
## Step2 DNN Model 
Pytorch is a neural network design module developed by Facebook. Import the module and its function to build dataset.
```pythonscript
import torch 
from torch.utils import data 
from torch.autograd import Variable 
import torchvision
from torchvision.datasets import mnist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
```
 DNN structure: 1 input layer(7 cells), 2 hiden layer(24 and 12 cells), 1 output layer(2 cells).

Activation function: leaky_relu

```pythonscript
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(7,24)
        self.fc2 = nn.Linear(24,12)
        self.fc3 = nn.Linear(12,2)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
Build dataset as pytorch module requied.

```pythonscript
class MyDataset(Dataset):
    def __init__(self,labels,features):
        super(MyDataset,self).__init__()
        self.labels=labels
        self.features=features
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self,idx):
        feature=self.features[idx]
        label=self.labels[idx]
        return {'feature':feature, 'label':label}
```
Define the training epoch.
```pythonscript
def train_epoch(loader):
    total_loss=0.0
    for i,data in enumerate(loader):
        featurest=data['feature'].float()
        labelst=data['label'].float()
        optimizer.zero_grad()
        out=mnet(featurest)
        loss=loss_fn(out,labelst)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print ('loss', total_loss/i)
```
## Step3 Training DNN Model 
Define the loss function as Mean Square Error (MSE).
Choose SGD as optimizer with 0.01 learning rate.
Momentum is a parameter to avoid local optimal.
Weight_decay is designed to avoid overfit.

```pythonscript
mnet=Net()
loss_fn=torch.nn.MSELoss()
optimizer=torch.optim.SGD(mnet.parameters(),lr=0.0001,momentum=0.01,weight_decay=1e-8)
```

Batch size represent the size of dataset each time provided for DNN to get loss and optimization.
The epoch time is 2000.
```pythonscript
dataset=MyDataset(np.asarray(labels_training),np.asarray(features_training))
load=DataLoader(dataset,shuffle=True,batch_size=100)
for epoch in range(2000):
    train_epoch(load)
```
Save the model after training done.
```pythonscript
torch.save(mnet.state_dict(),'DNN_model.pth')
```


# Part3 Test and Comparation
## Step1 Build Test Dataset
Test dataset was collected from the warframe market too. It contains 6 kinds of products with 535 rows of data.
Import the data from 'warframe_Test.csv' as table, and select 7 features to predict 2 labels.
```pythonscript
testing = pd.read_csv('Warframe_Test.csv')
Test=Table.from_df(testing)

features_test=[]
for i in range(534):
    feature1=[]
    feature1.append(Test.row(i)[7])
    feature1.append(Test.row(i)[8]/100)
    feature1.append(Test.row(i)[9])
    feature1.append(Test.row(i)[10])
    feature1.append(Test.row(i)[11]/100)
    feature1.append(Test.row(i)[12])
    feature1.append(Test.row(i)[13])
    features_test.append(feature1)

labels_test=[]
for i in range(534):
    label=[]
    label.append(Test.row(i)[1])
    label.append(Test.row(i)[4])
    labels_test.append(label)

```
## Step2 Test DNN Model
Import DNN model from the file we stored before.
```pythonscript
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(7,24) 
        self.fc2 = nn.Linear(24,12)
        self.fc3 = nn.Linear(12,2)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
mnet=Net()
mnet.load_state_dict(torch.load('DNNmodel.pth'))
```
Generate prediction of avg_price and the real avg_price as 2 arrays.
```pythonscript
prediction=np.array([])
real=np.array([])
j=1
for i in range(534):
    k=i
    state=features_test[k]
    innput=Variable(torch.FloatTensor(state))
    output=mnet(innput).detach().numpy()
    prediction=np.append(prediction,output[j])
    real=np.append(real,labels_test[k][j])
```
Select the prediction of product 'Frost' to show the plot of comparation.
```pythonscript
plt.title('DNN_Frost')
plt.xlabel('Date')
plt.ylabel('Avg_Price')

plt.plot(prediction,'g',linewidth = '2',label='prediction')
plt.xlim(89,177)
plt.plot(real,'r',linewidth = '2',label='real')
plt.ylim(0,400)

plt.legend()
plt.grid(True,linestyle = "-",color = 'gray' ,linewidth = '0.15',axis='both')
plt.show()
```

## Step3 Test SVR Model
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.
And SVR is one kind of SVM called Support Vector Machine for Regression. 
It is more like a linear regression way to do machine learning. 

We will use the result of SVR to compare the prediction from DNN. First, import the SVR module.
```pythonscript
from sklearn.svm import LinearSVR
from sklearn.datasets import make_regression
```
Generate the training set of SVR, and do the prediction.
```pythonscript
regr = LinearSVR(random_state=0, tol=1e-5)
features_training, labels_training = make_regression(n_features=7, random_state=0)
regr.fit(features_training, labels_training)

prediction_svr=regr.predict(features_test)
```
Show the plot of SVR's result.

```pythonscript
plt.title('SVR_Frost')
plt.xlabel('Date')
plt.ylabel('Avg_Price')

plt.plot(prediction_svr[:500],'g',linewidth = '2',label='prediction')
plt.xlim(89,177)
plt.plot(labels_test[:500],'r',linewidth = '2',label='real')
plt.ylim(0,400)

plt.legend()
plt.grid(True,linestyle = "-",color = 'gray' ,linewidth = '0.15',axis='both')
plt.show()
```
