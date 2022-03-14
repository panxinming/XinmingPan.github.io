---
layout: post
title: Final Project
---
We write this Reflection Blog Post as a group. Our group members are **Xinming Pan** and **Xinyu Dong**.


## Topics
- **Overall, what did you achieve in your project?**

>Our project is about breast cancer prediction. From our project, we learned how to use Python to do machine learning problems. What's more, We learned how to use pandas to do data cleaning, Plotly to do data visualization, and Flask to build WebApp. These are the most valuable thing I learned from PIC 16B and our Project. 
Besides these, we learned every algorithm may has bias, what we need to do is to prevent these bias and reduce them. Make our algorithm more justice.

- **What are two aspects of your project that you are especially proud of?**

>The **first** part we are proud of is the WebApp we created. Because we don't know how to apply machine learning into WebApp design, but finally we did it! We created a beautiful WebApp and our WebApp can run the code we written. And I think our App is very useful, it can help patients who has potential breast cancer to detect at home or online. The **second** part we are proud of is 


- **What are two things you would suggest doing to further improve your project? (You are not responsible for doing those things.)** 

>The **first** thing we need to improve is the decoration of our WebApp. Because we learned how to use CSS file to decorate our WebApp, but actually we did not try to do that. Our WebApp is not outstanding, if we want to attract more users, the decoration is necessary.
The **second** thing we want to improve is the bias. As we mention before, our App has some algorithm bias. Because all the data is collected comes from United States. So, if you are the patients who come from Aisa, Africa, Europe etc. The prediction may be incorrect. What we want to improve is to mention these bias before the user use our WebApp, let them know the bias.

- **How does what you achieved compare to what you set out to do in your proposal? (if you didn't complete everything in your proposal, that's fine!)**

>We almost done everything what we mentioned in our proposal. However, there are two thing we didn't complete. The first thing is we are going to save the data into database and use SQL command to extract our data initially, but finally we just use Pandas to load our data. The second thing we didn't complete is to use CSS file to decorate our WebApp. Because we learned how to use CSS file to decorate our WebApp, but actually we did not try to do that. Our WebApp is not outstanding, if we want to attract more users, the decoration is necessary.

- **What are three things you learned from the experience of completing your project? Data analysis techniques? Python packages? Git + GitHub? Etc?** 

>The three things we learned from the experience of completing our project are the use of Flask package, the use of TensorFlow  and machine learning knowledge.

- **How will your experience completing this project will help you in your future studies or career? Please be as specific as possible.**

> If we meet some intersting or useful knowledge associated with Python, we can write down them in our Blog. Yes, we are planning to use our Blog even though PIC 16B is finished. And WebApp construction is very useful in future career, because we use App every day. Basically, if we want to have a good career, we must build an ability to create WebApp. Now, we learned how to apply database, SQL and machine learning method into WebApp, that helps us to create more useful WebApp in the future.



## 1. First Step: Load Data
First, we are going to load our data and import all the packages we need.
```python
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import matplotlib.patches as mpatches 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
df = pd.read_csv("wdbc.csv")
```

Then, we write a funtion which can help us standarize and clean our data.
```python
def clean_data(df1):
    encoder = preprocessing.LabelEncoder().fit(df1['diagnosis'])
    df1['diagnosis'] = encoder.transform(df1['diagnosis'])
    df1 = df1.dropna()
    X = df1.drop(['diagnosis'],axis = 1)
    y = df1['diagnosis']
    #standarization
    X_mean=X.mean(axis=0)
    X_std=X.std(axis=0)
    X= (X-X_mean)/X_std
    df2 = pd.concat([y,X],axis=1)
    return df2  


dataset = clean_data(df)
```

Then, we split our data into training and testing.
```python
train,test=train_test_split(dataset,test_size=.3,random_state=42)

X_train=train.drop(['diagnosis'],axis = 1)
y_train=train['diagnosis']

X_test=test.drop(['diagnosis'],axis = 1)
y_test=test['diagnosis']
```

Then, we write two functions to prepare to find the confusion matrix and column score.
```python
def plot_confusionmatrix(model,Xt,yt):
    """
    Plots the confusion matrix of a classifier
    model:trainded classifiers
    Xt:test data
    yt:the true value of test 
    """
    y_model = model.predict(Xt) #the predicted value of test
    mat = confusion_matrix(yt,y_model) #compare the true value and the predicted value
    mat_df = pd.DataFrame(mat)
    #plot the confusion matrix
    sns.heatmap(mat,square = True,annot= True, fmt = 'd', cmap="YlGnBu", cbar = False)
    plt.xlabel("Predicted value")
    plt.ylabel("True value")    
    plt.title("confusion matrix")
    
def check_column_score(clf):
    """
    show the top n column combinations and their training score and testing score.
    clf:trained classifiers     
    """
    D = {}
    #get the training average accuracy of all the column combinations by cross_validation
    for i in range(len(cols)):
        col = cols[i]
        D[i] = cross_val_score(clf,X_train[col],y_train,cv=5).mean()
    L = list(D.items())
    #sort the training accuracy
    L.sort(key = lambda x:x[1],reverse = True)
    #get the top n column combinations
    best = L[0:7]
    for key,value in best:
        print(str(cols[key])+":")
        col = cols[key]
        clf.fit(X_train[col],y_train)
        j = clf.score(X_test[col],y_test).round(3)
        print(" Train score is:"+str(np.round(value,3)) + " --- Test score is:"+ str(j))
```

## 2. Second Step: Compare Models

### (a). Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
check_column_score(LR)
```
```
['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave.points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave.points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave.points_worst', 'symmetry_worst', 'fractal_dimension_worst']:
 Train score is:0.975 --- Test score is:0.982
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst', 'radius_worst', 'radius_mean', 'concavity_worst', 'concavity_mean']:
 Train score is:0.947 --- Test score is:0.965
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst']:
 Train score is:0.937 --- Test score is:0.965
['area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst']:
 Train score is:0.937 --- Test score is:0.959
['area_mean', 'concave.points_mean']:
 Train score is:0.912 --- Test score is:0.906
['radius_mean', 'perimeter_mean', 'area_mean', 'concave.points_mean', 'concavity_mean']:
 Train score is:0.91 --- Test score is:0.918
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean']:
 Train score is:0.904 --- Test score is:0.953
```

Get the accuracy and confusion matrix.
```python
LR.fit(X_train[cols5], y_train)
cv_score_test = cross_val_score(LR, X_test[cols5], y_test, cv=10).mean()
print(cv_score_test)
plot_confusionmatrix(LR,X_test[cols5],y_test)
```
```
0.9588235294117649
```
![confusion1.jpg]({{ site.baseurl }}/images/confusion1.png)

### (b). Dicision Tree
```python
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(max_depth = 12, criterion = 'entropy')
check_column_score(DT)
```
```
['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave.points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave.points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave.points_worst', 'symmetry_worst', 'fractal_dimension_worst']:
 Train score is:0.93 --- Test score is:0.947
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst']:
 Train score is:0.92 --- Test score is:0.959
['area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst']:
 Train score is:0.915 --- Test score is:0.947
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst', 'radius_worst', 'radius_mean', 'concavity_worst', 'concavity_mean']:
 Train score is:0.912 --- Test score is:0.942
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean']:
 Train score is:0.902 --- Test score is:0.924
['radius_mean', 'perimeter_mean', 'area_mean', 'concave.points_mean', 'concavity_mean']:
 Train score is:0.897 --- Test score is:0.924
['area_mean', 'concave.points_mean']:
 Train score is:0.895 --- Test score is:0.93
```

Get the accuracy and confusion matrix.

```python
DT.fit(X_train[cols5], y_train)
cv_score_test = cross_val_score(DT, X_test[cols5], y_test, cv=10).mean()
print(cv_score_test)
plot_confusionmatrix(DT,X_test[cols5],y_test)
```
```
0.9238562091503268
```
![confusion2.jpg]({{ site.baseurl }}/images/confusion2.png)


### (c). Neural Networks
```python
MLP = MLPClassifier(solver='adam', activation = 'relu', alpha=1e-5,max_iter=3000,
                     hidden_layer_sizes= (18,18,18),random_state=1)

check_column_score(MLP)
```
```
['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave.points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave.points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave.points_worst', 'symmetry_worst', 'fractal_dimension_worst']:
 Train score is:0.977 --- Test score is:0.982
['area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst']:
 Train score is:0.942 --- Test score is:0.971
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst', 'radius_worst', 'radius_mean', 'concavity_worst', 'concavity_mean']:
 Train score is:0.932 --- Test score is:0.965
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst']:
 Train score is:0.932 --- Test score is:0.959
['radius_mean', 'perimeter_mean', 'area_mean', 'concave.points_mean', 'concavity_mean']:
 Train score is:0.922 --- Test score is:0.959
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean']:
 Train score is:0.92 --- Test score is:0.959
['area_mean', 'concave.points_mean']:
 Train score is:0.917 --- Test score is:0.906
```

Get the accuracy and confusion matrix.
```python
MLP.fit(X_train[cols5], y_train)
cv_score_test = cross_val_score(MLP, X_test[cols5], y_test, cv=10).mean()
print(cv_score_test)
plot_confusionmatrix(MLP,X_test[cols5],y_test)
```
```
0.9591503267973855
```
![confusion3.jpg]({{ site.baseurl }}/images/confusion3.png)


### (d). Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
RF = RandomForestClassifier(n_estimators=100)
check_column_score(RF)
```
```
['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave.points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave.points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave.points_worst', 'symmetry_worst', 'fractal_dimension_worst']:
 Train score is:0.955 --- Test score is:0.965
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst']:
 Train score is:0.94 --- Test score is:0.965
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst', 'radius_worst', 'radius_mean', 'concavity_worst', 'concavity_mean']:
 Train score is:0.937 --- Test score is:0.959
['area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst']:
 Train score is:0.925 --- Test score is:0.959
['radius_mean', 'perimeter_mean', 'area_mean', 'concave.points_mean', 'concavity_mean']:
 Train score is:0.922 --- Test score is:0.912
['area_mean', 'concave.points_mean']:
 Train score is:0.922 --- Test score is:0.924
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean']:
 Train score is:0.922 --- Test score is:0.947
```

Get the accuracy and confusion matrix.
```python
RF.fit(X_train[cols5], y_train)
cv_score_test = cross_val_score(RF, X_test[cols5], y_test, cv=10).mean()
print(cv_score_test)
plot_confusionmatrix(RF,X_test[cols5],y_test)
```
```
0.9470588235294117
```
![confusion4.jpg]({{ site.baseurl }}/images/confusion4.png)


### (e). SVM
```python
SVM = SVC(kernel = 'rbf',C=1E6,gamma = 0.005)
check_column_score(SVM)
```
```
['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave.points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave.points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave.points_worst', 'symmetry_worst', 'fractal_dimension_worst']:
 Train score is:0.972 --- Test score is:0.947
['area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst']:
 Train score is:0.947 --- Test score is:0.971
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst', 'radius_worst', 'radius_mean', 'concavity_worst', 'concavity_mean']:
 Train score is:0.94 --- Test score is:0.93
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean', 'concave.points_mean', 'concave.points_worst']:
 Train score is:0.937 --- Test score is:0.947
['perimeter_worst', 'perimeter_mean', 'area_worst', 'area_mean']:
 Train score is:0.937 --- Test score is:0.977
['radius_mean', 'perimeter_mean', 'area_mean', 'concave.points_mean', 'concavity_mean']:
 Train score is:0.93 --- Test score is:0.971
['area_mean', 'concave.points_mean']:
 Train score is:0.912 --- Test score is:0.912
```

Get the accuracy and confusion matrix.
```python
SVM.fit(X_train[cols5], y_train)
cv_score_test = cross_val_score(SVM, X_test[cols5], y_test, cv=10).mean()
print(cv_score_test)
plot_confusionmatrix(SVM,X_test[cols5],y_test)
```
```
0.9594771241830065
```
![confusion5.jpg]({{ site.baseurl }}/images/confusion5.png)


### (f). Tensorflow
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
TF = tf.keras.models.Sequential([
    layers.Dense(100, input_shape = (30,), activation='relu'),
    layers.Dense(100,activation="sigmoid"),
    layers.Dense(10,activation="softmax"),
    layers.Dense(2)
])
```
```python
TF.summary()
```
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 100)               3100      
_________________________________________________________________
dense_1 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1010      
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 22        
=================================================================
Total params: 14,232
Trainable params: 14,232
Non-trainable params: 0
_________________________________________________________________
```

We train our models 100 times.
```python
# ready for training!
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
TF.compile(optimizer ="adam",
              loss = loss_fn,
              metrics = ["accuracy"])
# train them 100 times.
history = TF.fit(X_train[cols0], y_train, epochs = 100, verbose=1)
```
```
Epoch 1/100
13/13 [==============================] - 0s 539us/step - loss: 0.6700 - accuracy: 0.7563
Epoch 2/100
13/13 [==============================] - 0s 462us/step - loss: 0.5776 - accuracy: 0.8920
Epoch 3/100
13/13 [==============================] - 0s 692us/step - loss: 0.4587 - accuracy: 0.9296
Epoch 4/100
13/13 [==============================] - 0s 616us/step - loss: 0.4028 - accuracy: 0.9497
Epoch 5/100
13/13 [==============================] - 0s 539us/step - loss: 0.3747 - accuracy: 0.9648
Epoch 6/100
13/13 [==============================] - 0s 769us/step - loss: 0.3567 - accuracy: 0.9724
Epoch 7/100
13/13 [==============================] - 0s 462us/step - loss: 0.3421 - accuracy: 0.9774
Epoch 8/100
13/13 [==============================] - 0s 539us/step - loss: 0.3303 - accuracy: 0.9799
Epoch 9/100
13/13 [==============================] - 0s 539us/step - loss: 0.3191 - accuracy: 0.9799
Epoch 10/100
13/13 [==============================] - 0s 538us/step - loss: 0.3092 - accuracy: 0.9799
Epoch 11/100
13/13 [==============================] - 0s 769us/step - loss: 0.3005 - accuracy: 0.9799
Epoch 12/100
13/13 [==============================] - 0s 539us/step - loss: 0.2914 - accuracy: 0.9849
Epoch 13/100
13/13 [==============================] - 0s 539us/step - loss: 0.2826 - accuracy: 0.9849
Epoch 14/100
13/13 [==============================] - 0s 539us/step - loss: 0.2743 - accuracy: 0.9899
Epoch 15/100
13/13 [==============================] - 0s 539us/step - loss: 0.2674 - accuracy: 0.9899
Epoch 16/100
13/13 [==============================] - 0s 538us/step - loss: 0.2605 - accuracy: 0.9899
Epoch 17/100
13/13 [==============================] - 0s 462us/step - loss: 0.2541 - accuracy: 0.9925
Epoch 18/100
13/13 [==============================] - 0s 539us/step - loss: 0.2481 - accuracy: 0.9925
Epoch 19/100
13/13 [==============================] - 0s 692us/step - loss: 0.2424 - accuracy: 0.9925
Epoch 20/100
13/13 [==============================] - 0s 616us/step - loss: 0.2369 - accuracy: 0.9925
Epoch 21/100
13/13 [==============================] - 0s 615us/step - loss: 0.2316 - accuracy: 0.9925
Epoch 22/100
13/13 [==============================] - 0s 538us/step - loss: 0.2268 - accuracy: 0.9925
Epoch 23/100
13/13 [==============================] - 0s 538us/step - loss: 0.2218 - accuracy: 0.9925
Epoch 24/100
13/13 [==============================] - 0s 535us/step - loss: 0.2170 - accuracy: 0.9925
Epoch 25/100
13/13 [==============================] - 0s 539us/step - loss: 0.2140 - accuracy: 0.9925
Epoch 26/100
13/13 [==============================] - 0s 539us/step - loss: 0.2069 - accuracy: 0.9950
Epoch 27/100
13/13 [==============================] - 0s 693us/step - loss: 0.2024 - accuracy: 0.9950
Epoch 28/100
13/13 [==============================] - 0s 539us/step - loss: 0.1974 - accuracy: 0.9950
Epoch 29/100
13/13 [==============================] - 0s 539us/step - loss: 0.1928 - accuracy: 0.9975
Epoch 30/100
13/13 [==============================] - 0s 692us/step - loss: 0.1892 - accuracy: 0.9950
Epoch 31/100
13/13 [==============================] - 0s 501us/step - loss: 0.1850 - accuracy: 0.9975
Epoch 32/100
13/13 [==============================] - 0s 539us/step - loss: 0.1815 - accuracy: 0.9950
Epoch 33/100
13/13 [==============================] - 0s 539us/step - loss: 0.1774 - accuracy: 0.9975
Epoch 34/100
13/13 [==============================] - 0s 616us/step - loss: 0.1741 - accuracy: 0.9975
Epoch 35/100
13/13 [==============================] - 0s 693us/step - loss: 0.1707 - accuracy: 0.9975
Epoch 36/100
13/13 [==============================] - 0s 615us/step - loss: 0.1678 - accuracy: 0.9975
Epoch 37/100
13/13 [==============================] - 0s 462us/step - loss: 0.1648 - accuracy: 0.9975
Epoch 38/100
13/13 [==============================] - 0s 769us/step - loss: 0.1615 - accuracy: 0.9975
Epoch 39/100
13/13 [==============================] - 0s 539us/step - loss: 0.1586 - accuracy: 0.9975
Epoch 40/100
13/13 [==============================] - 0s 462us/step - loss: 0.1557 - accuracy: 0.9975
Epoch 41/100
13/13 [==============================] - 0s 538us/step - loss: 0.1530 - accuracy: 0.9975
Epoch 42/100
13/13 [==============================] - 0s 616us/step - loss: 0.1504 - accuracy: 0.9975
Epoch 43/100
13/13 [==============================] - 0s 846us/step - loss: 0.1478 - accuracy: 0.9975
Epoch 44/100
13/13 [==============================] - 0s 539us/step - loss: 0.1453 - accuracy: 0.9975
Epoch 45/100
13/13 [==============================] - 0s 539us/step - loss: 0.1430 - accuracy: 0.9975
Epoch 46/100
13/13 [==============================] - 0s 692us/step - loss: 0.1406 - accuracy: 0.9975
Epoch 47/100
13/13 [==============================] - 0s 539us/step - loss: 0.1383 - accuracy: 0.9975
Epoch 48/100
13/13 [==============================] - 0s 539us/step - loss: 0.1361 - accuracy: 0.9975
Epoch 49/100
13/13 [==============================] - 0s 539us/step - loss: 0.1339 - accuracy: 0.9975
Epoch 50/100
13/13 [==============================] - 0s 616us/step - loss: 0.1318 - accuracy: 0.9975
Epoch 51/100
13/13 [==============================] - 0s 692us/step - loss: 0.1298 - accuracy: 0.9975
Epoch 52/100
13/13 [==============================] - 0s 539us/step - loss: 0.1278 - accuracy: 0.9975
Epoch 53/100
13/13 [==============================] - 0s 616us/step - loss: 0.1258 - accuracy: 0.9975
Epoch 54/100
13/13 [==============================] - 0s 539us/step - loss: 0.1239 - accuracy: 0.9975
Epoch 55/100
13/13 [==============================] - 0s 539us/step - loss: 0.1220 - accuracy: 0.9975
Epoch 56/100
13/13 [==============================] - 0s 539us/step - loss: 0.1203 - accuracy: 0.9975
Epoch 57/100
13/13 [==============================] - 0s 539us/step - loss: 0.1185 - accuracy: 0.9975
Epoch 58/100
13/13 [==============================] - 0s 539us/step - loss: 0.1167 - accuracy: 0.9975
Epoch 59/100
13/13 [==============================] - 0s 616us/step - loss: 0.1151 - accuracy: 0.9975
Epoch 60/100
13/13 [==============================] - 0s 615us/step - loss: 0.1134 - accuracy: 0.9975
Epoch 61/100
13/13 [==============================] - 0s 539us/step - loss: 0.1118 - accuracy: 0.9975
Epoch 62/100
13/13 [==============================] - 0s 616us/step - loss: 0.1103 - accuracy: 0.9975
Epoch 63/100
13/13 [==============================] - 0s 538us/step - loss: 0.1087 - accuracy: 0.9975
Epoch 64/100
13/13 [==============================] - 0s 616us/step - loss: 0.1072 - accuracy: 0.9975
Epoch 65/100
13/13 [==============================] - 0s 539us/step - loss: 0.1058 - accuracy: 0.9975
Epoch 66/100
13/13 [==============================] - 0s 539us/step - loss: 0.1043 - accuracy: 0.9975
Epoch 67/100
13/13 [==============================] - 0s 539us/step - loss: 0.1030 - accuracy: 0.9975
Epoch 68/100
13/13 [==============================] - 0s 539us/step - loss: 0.1016 - accuracy: 0.9975
Epoch 69/100
13/13 [==============================] - 0s 462us/step - loss: 0.1003 - accuracy: 0.9975
Epoch 70/100
13/13 [==============================] - 0s 615us/step - loss: 0.0990 - accuracy: 0.9975
Epoch 71/100
13/13 [==============================] - 0s 692us/step - loss: 0.0977 - accuracy: 0.9975
Epoch 72/100
13/13 [==============================] - 0s 539us/step - loss: 0.0964 - accuracy: 0.9975
Epoch 73/100
13/13 [==============================] - 0s 539us/step - loss: 0.0952 - accuracy: 0.9975
Epoch 74/100
13/13 [==============================] - 0s 769us/step - loss: 0.0940 - accuracy: 0.9975
Epoch 75/100
13/13 [==============================] - 0s 539us/step - loss: 0.0928 - accuracy: 0.9975
Epoch 76/100
13/13 [==============================] - 0s 539us/step - loss: 0.0916 - accuracy: 0.9975
Epoch 77/100
13/13 [==============================] - 0s 616us/step - loss: 0.0905 - accuracy: 0.9975
Epoch 78/100
13/13 [==============================] - 0s 539us/step - loss: 0.0894 - accuracy: 0.9975
Epoch 79/100
13/13 [==============================] - 0s 616us/step - loss: 0.0883 - accuracy: 0.9975
Epoch 80/100
13/13 [==============================] - 0s 616us/step - loss: 0.0872 - accuracy: 0.9975
Epoch 81/100
13/13 [==============================] - 0s 539us/step - loss: 0.0862 - accuracy: 0.9975
Epoch 82/100
13/13 [==============================] - 0s 616us/step - loss: 0.0852 - accuracy: 0.9975
Epoch 83/100
13/13 [==============================] - 0s 616us/step - loss: 0.0842 - accuracy: 0.9975
Epoch 84/100
13/13 [==============================] - 0s 692us/step - loss: 0.0832 - accuracy: 0.9975
Epoch 85/100
13/13 [==============================] - 0s 539us/step - loss: 0.0822 - accuracy: 0.9975
Epoch 86/100
13/13 [==============================] - 0s 539us/step - loss: 0.0813 - accuracy: 0.9975
Epoch 87/100
13/13 [==============================] - 0s 769us/step - loss: 0.0803 - accuracy: 0.9975
Epoch 88/100
13/13 [==============================] - 0s 462us/step - loss: 0.0794 - accuracy: 0.9975
Epoch 89/100
13/13 [==============================] - 0s 539us/step - loss: 0.0785 - accuracy: 0.9975
Epoch 90/100
13/13 [==============================] - 0s 693us/step - loss: 0.0777 - accuracy: 0.9975
Epoch 91/100
13/13 [==============================] - 0s 539us/step - loss: 0.0768 - accuracy: 0.9975
Epoch 92/100
13/13 [==============================] - 0s 462us/step - loss: 0.0759 - accuracy: 0.9975
Epoch 93/100
13/13 [==============================] - 0s 539us/step - loss: 0.0751 - accuracy: 0.9975
Epoch 94/100
13/13 [==============================] - 0s 616us/step - loss: 0.0743 - accuracy: 0.9975
Epoch 95/100
13/13 [==============================] - 0s 692us/step - loss: 0.0735 - accuracy: 0.9975
Epoch 96/100
13/13 [==============================] - 0s 539us/step - loss: 0.0727 - accuracy: 0.9975
Epoch 97/100
13/13 [==============================] - 0s 539us/step - loss: 0.0719 - accuracy: 0.9975
Epoch 98/100
13/13 [==============================] - 0s 616us/step - loss: 0.0712 - accuracy: 0.9975
Epoch 99/100
13/13 [==============================] - 0s 539us/step - loss: 0.0704 - accuracy: 0.9975
Epoch 100/100
13/13 [==============================] - 0s 615us/step - loss: 0.0697 - accuracy: 0.9975
```

Then, plot the progress of the training over time:
```python
plt.plot(history.history["accuracy"])
plt.gca().set(xlabel = "epoch", ylabel = "training accuracy")
```

![history1.jpg]({{ site.baseurl }}/images/history1.png)

Finally, get our accuracy.
```python
TF.evaluate(X_test[cols0], y_test, verbose = 2)
```
```
[0.12011773139238358, 0.9707602262496948]
```

### (g). Simplify Our Model

Because our data has many columns, so we want to reduce dimensions and compare these models.

First, we create some models.
```python
cols0 = df.columns.values.tolist()
cols0 = cols0[2:32]
cols1 = ['perimeter_worst','perimeter_mean','area_worst','area_mean','concave.points_mean','concave.points_worst',
        'radius_worst','radius_mean','concavity_worst','concavity_mean']
cols2 = ["radius_mean","perimeter_mean","area_mean",'concave.points_mean','concavity_mean']
cols3 = ['perimeter_worst','perimeter_mean','area_worst','area_mean','concave.points_mean','concave.points_worst']
cols4 = ['perimeter_worst','perimeter_mean','area_worst','area_mean']
cols5 = ['area_worst','area_mean','concave.points_mean','concave.points_worst']
cols6 = ['area_mean','concave.points_mean']
cols =[cols0,cols1,cols2,cols3,cols4,cols5,cols6]
```

Plot the graph of accuracy of each model.
```python
df1 = pd.read_csv(".\\files\\train_score.csv")
df2 = pd.read_csv(".\\files\\test_score.csv")
fig, axes = plt.subplots(2, 1,figsize=(14,10)) 
x = [0, 1, 2, 3,4,5,6]
labels = ["cols0","cols1","cols2","cols3","cols4","cols5","cols6"]
df1.plot(kind='line',marker='o',title='Train Accuracy',ax=axes[0])
df2.plot(kind='line',marker='o',title='Test Accuracy',ax=axes[1]) 
axes[0].set_xticks(x, labels, rotation=30)
axes[1].set_xticks(x, labels, rotation=30)
plt.suptitle("The comparison of 6 Models")
```
![compare.jpg]({{ site.baseurl }}/images/compare.png)

## 3. Third Step: Create Our WebApp

We create a App python file and write down all the codes we need. Also, we write serval HTML pages, and we will show you later.

First, we want to show what's our App python code looks like.
```python
from flask import Flask, g, render_template, request, redirect,url_for
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


app = Flask(__name__)

# Create main page (fancy)

@app.route('/', methods=['POST','GET'])

def main():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('Preprocessing'))
    return render_template('main.html')

# Show url matching

@app.route('/Preprocessing/', methods=['POST','GET'])
def Preprocessing():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('Feature_selection'))
    return render_template('Preprocessing.html')

# Page with form
@app.route('/Feature_selection/', methods=['POST','GET'])
def Feature_selection():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('Models'))
    return render_template('Feature_selection.html')


@app.route('/Models/', methods=['POST', 'GET'])
def Models():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('Diagnosis'))
    return render_template('Models.html')


# File uploads and interfacing with complex Python
# basic version

@app.route('/Diagnosis/', methods=['POST', 'GET'])
def Diagnosis():
    if request.method == 'GET':
        return render_template('Diagnosis.html')
    else:
        try:
            # retrieve the image
            g.text1 = float(request.form['text1'])
            g.text2 = float(request.form['text2'])
            g.text3 = float(request.form['text3'])
            g.text4 = float(request.form['text4'])
            #standarize
            X_value = [g.text1,g.text2,g.text3,g.text4]
            df3 = pd.read_csv(".\\files\\standarize.csv")
            c=["mean","std"]
            df4 = df3.loc[:,c]
            X_value= (X_value-df4["mean"])/df4["std"]
            X_value=np.array(X_value).tolist()
            X_value = [X_value]

            g.m = request.form['model']
            if g.m == "LR":
                g.model = joblib.load(".\\models\\LR.m") 
            elif g.m == "DT":
                g.model = joblib.load(".\\models\\DT.m")
            elif g.m == "MLP":
                g.model = joblib.load(".\\models\\MLP.m")
            elif g.m == "RF":
                g.model = joblib.load(".\\models\\RF.m")
            elif g.m == "SVM":
                g.model = joblib.load(".\\models\\SVM.m")
            else:
                g.model = tf.keras.models.load_model(".\\models\\TF")
                        
            if g.m != 'TF':
                g.result = g.model.predict(X_value)[0]
            else:
                g.result = g.model.predict(X_value)[0][1]
            if g.result>0:
                r1 = 1
                r2 = 0
            else:
                r2 = 1
                r1 = 0
            return render_template('Diagnosis.html', result1 = r1, result2 = r2)
        except:
            return render_template('Diagnosis.html', error=True)
```

Then, we want to show you serval HTML raw code we created.

This is the preprocessing page.

```html
{%raw%}
{% extends 'base.html' %}

{% block header %}
  <h1>{% block title %}Preprocessing{% endblock %}</h1>
{% endblock %}

{% block content %}
<h2> Preprocessing </h2>
  <ul>
    <li>LabelEncoder</li>
    <li>Standarization</li>
  </ul>

<div align = center>
  <img src="https://raw.githubusercontent.com/xinyue-lily/xinyue-lily.github.io/master/images/analyse_scaled_ra.jpg" width="864px" height="360px">

  <p>  The comparison diagram of TNE before and after standarization</p>

  <img src="https://raw.githubusercontent.com/xinyue-lily/xinyue-lily.github.io/master/images/TSNE_TSNE_scaled.png" width="702px" height="280px">
</div>

<p></p>
<p></p>
<p></p>
<form method="post">
   <button type="submit">Next Page</button>
</form>

{% endblock %}
{%endraw%}
```

![preprocessing.jpg]({{ site.baseurl }}/images/preprocessing.png)

This is the models page.
```html
{%raw%}
{% extends 'base.html' %}

{% block header %}
  <h1>{% block title %}Models{% endblock %}</h1>
{% endblock %}

{% block content %}
<h2> Models </h2>
  <ul>
    <li>Logistic Regression</li>
    <li>Decision Tree</li>
    <li>Multilayer Perceptron</li>
    <li>Random Forest</li>
    <li>Support Vector Machine</li>
    <li>Tensor Flow</li>
  </ul>

  <h2> Features </h2>
  <ul>
    <li>cols0 = all features
    <li>cols1 = ['perimeter_worst','perimeter_mean','area_worst','area_mean','concave.points_mean','concave.points_worst',
                'radius_worst','radius_mean','concavity_worst','concavity_mean']</li> 
    <li>cols2 = ["radius_mean","perimeter_mean","area_mean",'concave.points_mean','concavity_mean']</li>
    <li>cols3 = ['perimeter_worst','perimeter_mean','area_worst','area_mean','concave.points_mean','concave.points_worst']</li>
    <li>cols4 = ['perimeter_worst','perimeter_mean','area_worst','area_mean']</li>
    <li>cols5 = ['area_worst','area_mean','concave.points_mean','concave.points_worst']</li>
    <li>cols6 = ['area_mean','concave.points_mean']</li>
  </ul>

<div align = center>
  <img src="https://raw.githubusercontent.com/xinyue-lily/xinyue-lily.github.io/master/images/models.png" width="826px" height="660px">

</div>

<p></p>
<p></p>
<p></p>
<form method="post">
   <button type="submit">Next Page</button>
</form>

{% endblock %}
{%endraw%}
```

![models.jpg]({{ site.baseurl }}/images/models.png)

## 4. Test Our WebApp

After, we finished our WebApp, we are going to test our WebApp to see if it works.

Here, we jump into the diagnosis page and enter the information of a potential breast cancer patient.

Finally, we get the prediction outcome.

![diagnosis1.jpg]({{ site.baseurl }}/images/diagnosis1.png)

![diagnosis2.jpg]({{ site.baseurl }}/images/diagnosis2.png)




> This is all about our project, if you want to get more information of our project and see how it works in detail. Please visit our Github Repository. Here is the link [https://github.com/panxinming/PIC16B-Proposal](https://github.com/panxinming/PIC16B-Proposal)
>
> ***Thank You!***