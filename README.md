# Data Sciecne Final Project

# Problem Description
I am now playing a game called 'warframe', and there is a trading webside for this game named 'warframe markect'(https://warframe.market/).

Suppose I want to buy a equipment called 'Volt Prime' from another players, as well as looking through the trading information on the website, using a scrapy seems to be a good choice to do so.

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

##Step2 Data Transform

The text is a json file. We should transform it to a dictionary to read the content. Let's import mudole to deal with it.

```pythonscript   
import json
import pandas as pd
from pandas.io.json import json_normalize
```

```pythonscript
data=json.loads(trade)
data1=data['payload']['orders']
```
So, here data1 is the dictionary I need as trading information. The next step is to read it and generate a table to store the things I am interested in.

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
Then, choose the information I need. Assume I want to know the player's gameID, the price, the region and so on, creat several arrays to store them.

```pythonscript
time=np.array([])
volume=np.array([])
min_price=np.array([])
max_price=np.array([])
avg_price=np.array([])
median_price=np.array([])
warframe=np.array([])
```
Read the dictionary to input elements. Here I plan to get 400 orders.
```pythonscript
for i in range (0,89):
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
At last, store the talbe as a csv file.

```pythonscript
trading.to_csv('Volt Prime.csv')
```

## Step3 Data Organization

Now, I want to know the orders of seller. Thus, we can build a new table to select the rows needed. 
```pythonscript
import glob
```
or I would like to know the sellers that are online right now.

```pythonscript
csvx_list = glob.glob('*.csv')
for i in csvx_list:
    fr = open(i,'r').read()
    with open('training_data.csv','a') as f:
        f.write(fr)
```


# Part2 Nerual Network Model
## Step1 Build Training Dateset
```pythonscript
training = pd.read_csv('warframe_data.csv')
Train=Table.from_df(training)
Train
```

training table

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


```pythonscript
labels_training=[]
for i in range(1335):
    label=[]
    label.append(Train.row(i)[1])
    label.append(Train.row(i)[4])
    labels_training.append(label)
```
## Step2 DNN Model 

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
```pythonscript
mnet=Net()
loss_fn=torch.nn.MSELoss()
optimizer=torch.optim.SGD(mnet.parameters(),lr=0.0001,momentum=0.01,weight_decay=1e-8)

dataset=MyDataset(np.asarray(labels_training),np.asarray(features_training))
load=DataLoader(dataset,shuffle=True,batch_size=100)

for epoch in range(2000):
    train_epoch(load)
```


```pythonscript
torch.save(mnet.state_dict(),'DNN_model.pth')
```


# Part3 Test and Comparation
## Step1 Build Test Dataset
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

```pythonscript
from sklearn.svm import LinearSVR
from sklearn.datasets import make_regression
```

```pythonscript
regr = LinearSVR(random_state=0, tol=1e-5)
features_training, labels_training = make_regression(n_features=7, random_state=0)
regr.fit(features_training, labels_training)

prediction_svr=regr.predict(features_test)
```

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
