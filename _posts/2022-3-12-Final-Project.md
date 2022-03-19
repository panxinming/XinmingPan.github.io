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

>The **first** part we are proud of is the WebApp we created. Because we don't know how to apply machine learning into WebApp design, but finally we did it! We created a beautiful WebApp and our WebApp can run the code we written. And I think our App is very useful, it can help patients who has potential breast cancer to detect at home or online. The **second** part we are proud of is the dimension reduction, we learned PCA in UCLA, but here we want to use another dimension reduction method which is T-SNE, we searched in Google and ask for someone else to explain how to apply T-SNE in Python. Finally, we got this. So, we are very proud of the dimension reduction we did.


- **What are two things you would suggest doing to further improve your project? (You are not responsible for doing those things.)** 

>The **first** thing we need to improve is the decoration of our WebApp. Because we learned how to use CSS file to decorate our WebApp, but actually we did not try to do that. Our WebApp is not outstanding, if we want to attract more users, the decoration is necessary.
The **second** thing we want to improve is the bias. As we mention before, our App has some algorithm bias. Because all the data is collected comes from United States. So, if you are the patients who come from Aisa, Africa, Europe etc. The prediction may be incorrect. What we want to improve is to mention these bias before the user use our WebApp, let them know the bias.

- **How does what you achieved compare to what you set out to do in your proposal? (if you didn't complete everything in your proposal, that's fine!)**

>We almost done everything what we mentioned in our proposal. However, there are two thing we didn't complete. The first thing is we are going to save the data into database and use SQL command to extract our data initially, but finally we just use Pandas to load our data. The second thing we didn't complete is to use CSS file to decorate our WebApp. Because we learned how to use CSS file to decorate our WebApp, but actually we did not try to do that. Our WebApp is not outstanding, if we want to attract more users, the decoration is necessary.

- **What are three things you learned from the experience of completing your project? Data analysis techniques? Python packages? Git + GitHub? Etc?** 

>The three things we learned from the experience of completing our project are the use of Flask package, the use of TensorFlow  and machine learning knowledge.

- **How will your experience completing this project will help you in your future studies or career? Please be as specific as possible.**

> If we meet some intersting or useful knowledge associated with Python, we can write down them in our Blog. Yes, we are planning to use our Blog even though PIC 16B is finished. And WebApp construction is very useful in future career, because we use App every day. Basically, if we want to have a good career, we must build an ability to create WebApp. Now, we learned how to apply database, SQL and machine learning method into WebApp, that helps us to create more useful WebApp in the future.

## Project Structure
**The project is divided into three parts:**

- Data analysis and feature selection

- Model training and comparison

- The webapp of diagnosis

## 1. Data analysis and feature selection
First, we are going to load our data and use pairplot to visualize the relationship between the top ten features.
```python
df = pd.read_csv("wdbc.csv")
cols = df.columns.values.tolist()
cols = cols[1:12]
sns.pairplot(df[cols],hue='diagnosis', height=1.5)
```
![pairplot.png]({{ site.baseurl }}/images/pairplot.png)

The plots on the diagonal line describes the distribution of individual features in each category, and the plots in the upper triangle and the lower triangle illustrate the pairwise relationships between different features. It is obvious that there is strong correlation between the radius, the perimeter, and the area. In data analysis we only need to keep one of such features so that the features are independent to one another. By observation, we can also know that there is a linear relationship between the concavity and the concave point, and between the concavity and the compactness.

### (1) Standardization
The data type of tag diagnosis is character, "B" indicates benign and "m" indicates malignant. Use labelencoder to digitize it and encode the tag value as 0,1, so that it can be used as the training tag of the model.

Every original data set may have differences in orders in dimension and different magnitudes. If we do not make modifications to them and use the original value directly, this will highlight the role of the features with higher numerical value in the analysis and weaken the role of the features with lower numerical value. To ensure the reliability of our results, we need to standardize the original data. After standardization, every feature has a mean of 0 and a standard deviation of 1.

```python
encoder = preprocessing.LabelEncoder().fit(df['diagnosis'])
df['diagnosis'] = encoder.transform(df['diagnosis'])
X = df.drop(['diagnosis','Ob'],axis = 1)
y = df['diagnosis']
X_mean=X.mean(axis=0)
X_std=X.std(axis=0)
X_scaled= (X-X_mean)/X_std
```
We drawed TSNE diagrams for the original dataset and the dataset after standardization. 
```python
f, axes = plt.subplots(1, 2, figsize=(12,4))
tsne = TSNE(n_components=2, perplexity = 50)
dataset_tsne = tsne.fit_transform(X)
dataset_scaled = tsne.fit_transform(X_scaled)
ax1 = axes[0].scatter(dataset_tsne[:, 0], dataset_tsne[:, 1], c = y, alpha = 0.7,cmap="jet")
ax2 = axes[1].scatter(dataset_scaled[:, 0], dataset_scaled[:, 1], c = y, alpha = 0.7,cmap="jet")
axes[0].set(title = "Orignal Dataset")
axes[1].set(title = "Scaled Dataset")
f.colorbar(ax1, ax = axes[0])
f.colorbar(ax2, ax = axes[1])
plt.suptitle('perplexity = 50',ha='center')
```
![TSNE.png]({{ site.baseurl }}/images/TSNE_TSNE_scaled.png)

We can see from the plot that the sample separation has greatly improved.

Next, We choose six features, including two mean features, two variance features and two maximum features. These three types of data are different in terms of properties and magnitudes. Before we standardize the features of datasets, we can observe, as shown in the graph on the left, that the standard deviation feature value distribution is concentrated in a small interval with a small value. The mean feature distribution, however, is wide and the value is large. As shown in the right figure. After standardization, the distribution of features falls into the same interval, which ensures the reliability of classification.

```python
# Draw the comparison diagram of KDE before and after standardization
features = df.columns.values.tolist()
features = features[2:32]
fig,axes = plt.subplots(1, 2, figsize = (12, 5), sharex=False)
for i in [0, 1, 10, 11, 20, 21]: 
    sns.kdeplot(X.iloc[:,i], label=features[i], shade=True, ax=axes[0])
    sns.kdeplot(X_scaled.iloc[:, i], label=features[i], shade=True, ax=axes[1])
axes[0].set(title = "Orignal Dataset")
axes[1].set(title = "Scaled Dataset")
axes[0].legend()
axes[1].legend()
plt.suptitle('The comparison diagram of KDE before and after standardization',ha='center')
```
![kde.png]({{ site.baseurl }}/images/kde.png)


### (2) Feature Selection
Feature selection is the process of selecting the most effective features form a pool of original features to reduce data dimension. Since this dataset has many features, feature selection is a good way to improve the performance of the machine learning algorithm.

SelectKBest selects the best characteristics and uses analysis of variance to calculate feature scores. In general, variance analysis is to test whether a random variable has gone through significant changes if it is tested under different levels. This is used to test the correlation between two variables. 

```python
from sklearn.feature_selection import SelectKBest,f_classif
classifer = SelectKBest(f_classif, k=10).fit(X_scaled, y)
mask = classifer.get_support()
X_new = classifer.transform(X)
features = df.columns.values.tolist()[2:]
selected = []
for feature, b in zip(features, mask):  
    if b == True:
        selected.append(feature)
print(selected)
```
```python
['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 
'concave.points_mean', 'radius_worst', 'perimeter_worst', 'area_worst', 
'concavity_worst', 'concave.points_worst']
```
It is clear that the mean and maximum values of the radius, perimeter, area, concavity, and concave points are the best features.

## 2. Model training and comparison

### (1) Data Preparation
Firstly, We split the dataset into the test dataset and the train dataset. The train dataset takes up 70% of the entire data, and the test dataset takes up the rest 30%.
```python
dataset = pd.concat([y,X],axis=1)
train,test=train_test_split(dataset,test_size=.3,random_state=42)

X_train=train.drop(['diagnosis'],axis = 1)
y_train=train['diagnosis']
X_test=test.drop(['diagnosis'],axis = 1)
y_test=test['diagnosis']
```
we choose seven different combinations of columns and use them in the comparative experiments.

- ***cols0***: all the ***30*** features combination.

- ***cols1***: ***10*** features combination using *selectKBest*.

- ***cols2***: ***5*** mean features combination: *radius_mean,perimeter_mean,area_mean,concave.points_mean and concavity_mean*.

- ***cols3***: ***6*** features combination: *perimeter_worst,perimeter_mean,area_worst,area_mean,concave.points_mean and concave.points_worst*.

- ***cols4***: ***4*** features combination: *perimeter_worst,perimeter_mean,area_worst and area_mean*.

- ***cols5***: ***4*** features combination: *area_worst,area_mean,concave.points_mean and concave.points_worst*.

- ***cols6***: ***2*** features combination: *area_mean and concave.points_mean*.

```python
cols0 = df.columns.values.tolist()
cols0 = cols0[2:32]
cols1 = ['perimeter_worst','perimeter_mean','area_worst','area_mean',
         'concave.points_mean','concave.points_worst','radius_worst',
         'radius_mean','concavity_worst','concavity_mean']
cols2 = ['radius_mean','perimeter_mean','area_mean','concave.points_mean',
         'concavity_mean']
cols3 = ['perimeter_worst','perimeter_mean','area_worst','area_mean',
         'concave.points_mean','concave.points_worst']
cols4 = ['perimeter_worst','perimeter_mean','area_worst','area_mean']
cols5 = ['area_worst','area_mean','concave.points_mean','concave.points_worst']
cols6 = ['area_mean','concave.points_mean']
cols =[cols0,cols1,cols2,cols3,cols4,cols5,cols6]
```
### (2) Two Functions

- plot_confusionmatrix：plots the confusion matrix of a model.

- check_column_score：sorts the training average accuracy of all the feature combinations by cross validation, then show the top n column combinations and their train score and test score.

```python
def plot_confusionmatrix(model,Xt,yt):
    """
    Plots the confusion matrix of a classifier
    model:trainded classifiers
    Xt:test data
    yt:the true value of test 
    """
    y_model = model.predict(Xt) #the predicted value of test
    #compare the true value and the predicted value
    mat = confusion_matrix(yt,y_model) 
    mat_df = pd.DataFrame(mat)
    #plot the confusion matrix
    sns.heatmap(mat,square = True,annot= True, fmt = 'd', 
                cmap="YlGnBu", cbar = False)
    plt.xlabel("Predicted value")
    plt.ylabel("True value")    
    plt.title("confusion matrix")
    
def check_column_score(clf):
    """
    show the top n column combinations and their training score 
    and testing score.
    clf:trained classifiers     
    """
    D = {}
    #get the training average accuracy of all the column 
    #combinations by cross_validation
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
        print(" Train score is:"+str(np.round(value,3)) +
              " --- Test score is:"+ str(j))
```

### (3) Train and save the models
#### (a) Logistic Regression
We build a logistic regression model and use all the possible feature combinations of this model to do the training and testing. We calculate the training and testing accuracies and order them by training accuracy.
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

We choose the features in cols5 and train the model based on those features. Then we test the accuracy on the testing dataset and plot the confusion matrix with the function plot_confusionmatrix.
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

Save the trained logistic regression model in .\models\LR.m so that it can be called in webapp.

```python
joblib.dump(LR, ".\\models\\LR.m")
``` 

#### (b) Decision Tree
We build a decision tree model and use all the possible feature combinations of this model to do the training and testing. We calculate the training and testing accuracies and order them by training accuracy.
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

We choose the features in cols5 and train the model based on those features. Then we test the accuracy on the testing dataset and plot the confusion matrix with the function plot_confusionmatrix.

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

Save the trained decision tree model in .\models\DT.m so that it can be called in webapp.

```python
joblib.dump(DT, ".\\models\\DT.m")
``` 

#### (c) Multilayer Perceptron
We build a multilayer perceptron model and use all the possible feature combinations of this model to do the training and testing. We calculate the training and testing accuracies and order them by training accuracy.
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

We choose the features in cols5 and train the model based on those features. Then we test the accuracy on the testing dataset and plot the confusion matrix with the function plot_confusionmatrix.
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

Save the trained multilayer perceptron model in .\models\MLP.m so that it can be called in webapp.

```python
joblib.dump(MLP, ".\\models\\MLP.m")
``` 

#### (d) Random Forest
We build a random forest model and use all the possible feature combinations of this model to do the training and testing. We calculate the training and testing accuracies and order them by training accuracy.
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

We choose the features in cols5 and train the model based on those features. Then we test the accuracy on the testing dataset and plot the confusion matrix with the function plot_confusionmatrix.
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

Save the trained random forest model in .\models\RF.m so that it can be called in webapp.

```python
joblib.dump(RF, ".\\models\\RF.m")
``` 

#### (e) SVM
We build a support vector machine model and use all the possible feature combinations of this model to do the training and testing. We calculate the training and testing accuracies and order them by training accuracy.
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

We choose the features in cols5 and train the model based on those features. Then we test the accuracy on the testing dataset and plot the confusion matrix with the function plot_confusionmatrix.
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

Save the trained support vector machine model in .\models\SVM.m so that it can be called in webapp.

```python
joblib.dump(SVM, ".\\models\\SVM.m")
``` 

#### (f) Tensorflow
We build a Tensorflow model with the four features of cols5. 
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
TF1 = tf.keras.models.Sequential([
    layers.Dense(100, input_shape = (4,), activation='relu'),
    layers.Dense(100,activation="sigmoid"),
    layers.Dense(10,activation="softmax"),
    layers.Dense(2)    
])
# ready for training!
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
TF1.compile(optimizer ="adam",
              loss = loss_fn,
              metrics = ["accuracy"])
# train them 100 times.
history = TF1.fit(X_train[cols5], y_train, epochs = 100, verbose=1)
```

```python
Epoch 1/100
13/13 [================] - 0s 2ms/step - loss: 0.6619 - accuracy: 0.6307
Epoch 2/100
13/13 [================] - 0s 2ms/step - loss: 0.6089 - accuracy: 0.6759
Epoch 3/100
13/13 [================] - 0s 2ms/step - loss: 0.5351 - accuracy: 0.8367
Epoch 4/100
13/13 [================] - 0s 2ms/step - loss: 0.4581 - accuracy: 0.9121
Epoch 5/100
13/13 [================] - 0s 3ms/step - loss: 0.4114 - accuracy: 0.9296
Epoch 6/100
13/13 [================] - 0s 4ms/step - loss: 0.3895 - accuracy: 0.9322
Epoch 7/100
13/13 [================] - 0s 3ms/step - loss: 0.3766 - accuracy: 0.9322
Epoch 8/100
13/13 [================] - 0s 2ms/step - loss: 0.3667 - accuracy: 0.9397
Epoch 9/100
13/13 [================] - 0s 2ms/step - loss: 0.3583 - accuracy: 0.9422
Epoch 10/100
13/13 [================] - 0s 3ms/step - loss: 0.3511 - accuracy: 0.9422
Epoch 11/100
13/13 [================] - 0s 2ms/step - loss: 0.3446 - accuracy: 0.9422
Epoch 12/100
13/13 [================] - 0s 2ms/step - loss: 0.3384 - accuracy: 0.9372
Epoch 13/100
13/13 [================] - 0s 2ms/step - loss: 0.3322 - accuracy: 0.9397
Epoch 14/100
13/13 [================] - 0s 2ms/step - loss: 0.3270 - accuracy: 0.9397
Epoch 15/100
13/13 [================] - 0s 2ms/step - loss: 0.3216 - accuracy: 0.9422
Epoch 16/100
13/13 [================] - 0s 2ms/step - loss: 0.3167 - accuracy: 0.9397
Epoch 17/100
13/13 [================] - 0s 2ms/step - loss: 0.3120 - accuracy: 0.9397
Epoch 18/100
13/13 [================] - 0s 2ms/step - loss: 0.3068 - accuracy: 0.9397
Epoch 19/100
13/13 [================] - 0s 2ms/step - loss: 0.3024 - accuracy: 0.9422
Epoch 20/100
13/13 [================] - 0s 2ms/step - loss: 0.2982 - accuracy: 0.9397
Epoch 21/100
13/13 [================] - 0s 2ms/step - loss: 0.2951 - accuracy: 0.9422
Epoch 22/100
13/13 [================] - 0s 2ms/step - loss: 0.2894 - accuracy: 0.9397
Epoch 23/100
13/13 [================] - 0s 2ms/step - loss: 0.2867 - accuracy: 0.9397
Epoch 24/100
13/13 [================] - 0s 2ms/step - loss: 0.2816 - accuracy: 0.9397
Epoch 25/100
13/13 [================] - 0s 3ms/step - loss: 0.2772 - accuracy: 0.9372
Epoch 26/100
13/13 [================] - 0s 2ms/step - loss: 0.2743 - accuracy: 0.9422
Epoch 27/100
13/13 [================] - 0s 2ms/step - loss: 0.2704 - accuracy: 0.9397
Epoch 28/100
13/13 [================] - 0s 2ms/step - loss: 0.2660 - accuracy: 0.9422
Epoch 29/100
13/13 [================] - 0s 2ms/step - loss: 0.2629 - accuracy: 0.9422
Epoch 30/100
13/13 [================] - 0s 2ms/step - loss: 0.2586 - accuracy: 0.9447
Epoch 31/100
13/13 [================] - 0s 2ms/step - loss: 0.2551 - accuracy: 0.9447
Epoch 32/100
13/13 [================] - 0s 2ms/step - loss: 0.2513 - accuracy: 0.9472
Epoch 33/100
13/13 [================] - 0s 2ms/step - loss: 0.2478 - accuracy: 0.9497
Epoch 34/100
13/13 [================] - 0s 2ms/step - loss: 0.2441 - accuracy: 0.9472
Epoch 35/100
13/13 [================] - 0s 2ms/step - loss: 0.2421 - accuracy: 0.9523
Epoch 36/100
13/13 [================] - 0s 2ms/step - loss: 0.2370 - accuracy: 0.9548
Epoch 37/100
13/13 [================] - 0s 2ms/step - loss: 0.2341 - accuracy: 0.9598
Epoch 38/100
13/13 [================] - 0s 2ms/step - loss: 0.2317 - accuracy: 0.9598
Epoch 39/100
13/13 [================] - 0s 2ms/step - loss: 0.2286 - accuracy: 0.9598
Epoch 40/100
13/13 [================] - 0s 2ms/step - loss: 0.2266 - accuracy: 0.9623
Epoch 41/100
13/13 [================] - 0s 2ms/step - loss: 0.2220 - accuracy: 0.9598
Epoch 42/100
13/13 [================] - 0s 2ms/step - loss: 0.2201 - accuracy: 0.9598
Epoch 43/100
13/13 [================] - 0s 2ms/step - loss: 0.2174 - accuracy: 0.9673
Epoch 44/100
13/13 [================] - 0s 2ms/step - loss: 0.2154 - accuracy: 0.9623
Epoch 45/100
13/13 [================] - 0s 2ms/step - loss: 0.2117 - accuracy: 0.9623
Epoch 46/100
13/13 [================] - 0s 2ms/step - loss: 0.2092 - accuracy: 0.9623
Epoch 47/100
13/13 [================] - 0s 2ms/step - loss: 0.2063 - accuracy: 0.9623
Epoch 48/100
13/13 [================] - 0s 2ms/step - loss: 0.2043 - accuracy: 0.9648
Epoch 49/100
13/13 [================] - 0s 2ms/step - loss: 0.2018 - accuracy: 0.9648
Epoch 50/100
13/13 [================] - 0s 2ms/step - loss: 0.1998 - accuracy: 0.9648
Epoch 51/100
13/13 [================] - 0s 2ms/step - loss: 0.1979 - accuracy: 0.9673
Epoch 52/100
13/13 [================] - 0s 2ms/step - loss: 0.1966 - accuracy: 0.9673
Epoch 53/100
13/13 [================] - 0s 2ms/step - loss: 0.1941 - accuracy: 0.9648
Epoch 54/100
13/13 [================] - 0s 2ms/step - loss: 0.1912 - accuracy: 0.9673
Epoch 55/100
13/13 [================] - 0s 2ms/step - loss: 0.1906 - accuracy: 0.9648
Epoch 56/100
13/13 [================] - 0s 2ms/step - loss: 0.1880 - accuracy: 0.9673
Epoch 57/100
13/13 [================] - 0s 2ms/step - loss: 0.1858 - accuracy: 0.9698
Epoch 58/100
13/13 [================] - 0s 2ms/step - loss: 0.1854 - accuracy: 0.9673
Epoch 59/100
13/13 [================] - 0s 2ms/step - loss: 0.1852 - accuracy: 0.9648
Epoch 60/100
13/13 [================] - 0s 2ms/step - loss: 0.1811 - accuracy: 0.9698
Epoch 61/100
13/13 [================] - 0s 2ms/step - loss: 0.1797 - accuracy: 0.9698
Epoch 62/100
13/13 [================] - 0s 2ms/step - loss: 0.1775 - accuracy: 0.9698
Epoch 63/100
13/13 [================] - 0s 3ms/step - loss: 0.1753 - accuracy: 0.9673
Epoch 64/100
13/13 [================] - 0s 2ms/step - loss: 0.1737 - accuracy: 0.9673
Epoch 65/100
13/13 [================] - 0s 2ms/step - loss: 0.1732 - accuracy: 0.9698
Epoch 66/100
13/13 [================] - 0s 2ms/step - loss: 0.1706 - accuracy: 0.9698
Epoch 67/100
13/13 [================] - 0s 2ms/step - loss: 0.1704 - accuracy: 0.9698
Epoch 68/100
13/13 [================] - 0s 2ms/step - loss: 0.1689 - accuracy: 0.9698
Epoch 69/100
13/13 [================] - 0s 2ms/step - loss: 0.1666 - accuracy: 0.9724
Epoch 70/100
13/13 [================] - 0s 2ms/step - loss: 0.1651 - accuracy: 0.9698
Epoch 71/100
13/13 [================] - 0s 2ms/step - loss: 0.1650 - accuracy: 0.9724
Epoch 72/100
13/13 [================] - 0s 2ms/step - loss: 0.1639 - accuracy: 0.9673
Epoch 73/100
13/13 [================] - 0s 2ms/step - loss: 0.1609 - accuracy: 0.9724
Epoch 74/100
13/13 [================] - 0s 2ms/step - loss: 0.1611 - accuracy: 0.9698
Epoch 75/100
13/13 [================] - 0s 2ms/step - loss: 0.1582 - accuracy: 0.9724
Epoch 76/100
13/13 [================] - 0s 3ms/step - loss: 0.1579 - accuracy: 0.9724
Epoch 77/100
13/13 [================] - 0s 2ms/step - loss: 0.1568 - accuracy: 0.9724
Epoch 78/100
13/13 [================] - 0s 2ms/step - loss: 0.1553 - accuracy: 0.9724
Epoch 79/100
13/13 [================] - 0s 3ms/step - loss: 0.1530 - accuracy: 0.9749
Epoch 80/100
13/13 [================] - 0s 3ms/step - loss: 0.1529 - accuracy: 0.9749
Epoch 81/100
13/13 [================] - 0s 4ms/step - loss: 0.1515 - accuracy: 0.9774
Epoch 82/100
13/13 [================] - 0s 2ms/step - loss: 0.1501 - accuracy: 0.9724
Epoch 83/100
13/13 [================] - 0s 2ms/step - loss: 0.1515 - accuracy: 0.9724
Epoch 84/100
13/13 [================] - 0s 2ms/step - loss: 0.1510 - accuracy: 0.9724
Epoch 85/100
13/13 [================] - 0s 2ms/step - loss: 0.1470 - accuracy: 0.9774
Epoch 86/100
13/13 [================] - 0s 2ms/step - loss: 0.1458 - accuracy: 0.9749
Epoch 87/100
13/13 [================] - 0s 2ms/step - loss: 0.1457 - accuracy: 0.9724
Epoch 88/100
13/13 [================] - 0s 3ms/step - loss: 0.1439 - accuracy: 0.9724
Epoch 89/100
13/13 [================] - 0s 2ms/step - loss: 0.1462 - accuracy: 0.9724
Epoch 90/100
13/13 [================] - 0s 2ms/step - loss: 0.1455 - accuracy: 0.9724
Epoch 91/100
13/13 [================] - 0s 2ms/step - loss: 0.1416 - accuracy: 0.9774
Epoch 92/100
13/13 [================] - 0s 2ms/step - loss: 0.1412 - accuracy: 0.9774
Epoch 93/100
13/13 [================] - 0s 2ms/step - loss: 0.1387 - accuracy: 0.9774
Epoch 94/100
13/13 [================] - 0s 2ms/step - loss: 0.1387 - accuracy: 0.9724
Epoch 95/100
13/13 [================] - 0s 2ms/step - loss: 0.1372 - accuracy: 0.9774
Epoch 96/100
13/13 [================] - 0s 2ms/step - loss: 0.1370 - accuracy: 0.9774
Epoch 97/100
13/13 [================] - 0s 2ms/step - loss: 0.1365 - accuracy: 0.9749
Epoch 98/100
13/13 [================] - 0s 2ms/step - loss: 0.1358 - accuracy: 0.9774
Epoch 99/100
13/13 [================] - 0s 2ms/step - loss: 0.1344 - accuracy: 0.9749
Epoch 100/100
13/13 [================] - 0s 2ms/step - loss: 0.1332 - accuracy: 0.9774
```

Then, plot the progress of the training over time:
```python
plt.plot(history.history["accuracy"])
plt.gca().set(xlabel = "epoch", ylabel = "training accuracy")
```

![history1.jpg]({{ site.baseurl }}/images/history2.png)

Finally, get the testing accuracy.
```python
TF1.evaluate(X_test[cols5], y_test, verbose = 2)
```
```python
[0.12011773139238358, 0.9707602262496948]
```
Save the trained Tensorflow model in models\TF and use it in webapp.
```python
TF1.save('.\\models\\TF') 
```

#### (g) Comparison of model accuracy

Plot the graph of the training accuracy and testing accuracy of each model.
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

### My Oberservations:
- We can observe from the plot that the set containing all features has the highest accuracy in training and testing. The testing accuracy, 98.6%, is the highest using the logistic regression model for this set. cols1 used the 10 best features we selected using SelectKBest, and the testing accuracy is 97.06% using tensorflow. We chose 6 distinct features from cols1 to form cols3, which also has a high testing accuracy.

- Cols4 is a subset of the set of the 10 best features we selected using SelectKBest which consists of the features perimeter_mean, perimeter_worst,area_mean and area_worst. From what we analyzed before we know that the perimeter and the area of a cell has high correlations with each other, so the testing accuracy of cols4 is a bit low, but its testing accuracy is very high. 

- For cols5 we chose the features the area_mean,area_worst,concave.points_mean and concave.points_worst. The area of the cell and the number of its concave points have no direct correlations with each other, and they are vastly different in terms of their order of magnitude. However, after standardization, this set of features achieved very high accuracy in both testing and training. In our experiment we can see that this set of features achieved the second or third highest in accuracy using most of our machine learning models.

- Cols2 chose all the mean features in cols1 but diacarded all the worst features. Cols6 only chose 2 features. Compared to other column combinations, their results are a bit lower in accuracy.

- In these six distinct machine learning algorithms, we can get the tensorflow has the highest accuracy overall, with an accuracy of 97.66% with cols0, and an accuracy of 97.07% with cols5.

- All in all, although using all the 30 features achieves the highest testing accuracy, in practice it is too troublesome to collect 30 features one by one. Thus we choose to use col5, which only chose 4 out of the 30 columns but still achieves high testing accuracy. In our webapp, we only type in the 4 features of cols5 and use it to make predictions using our trained machine learning algorithms. 

## 3. Create Our WebApp

We create a App python file and write down all the codes we need. Also, we write serval HTML pages, and we will show you later.

First, we want to show what's our App python code looks like.
```python
from flask import Flask, g, render_template, request, redirect,url_for
import pickle
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

app = Flask(__name__)

# The function gives us the main page of the Webapp.
@app.route('/', methods=['POST','GET'])
def main():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('Preprocessing'))
    return render_template('main.html')

# The function gives us the preprecessing page of the Webapp.
# Call the template preprecssing.html and show preprocessing page.
# If we click on the "next" button, We will jump to feature_selection page
@app.route('/Preprocessing/', methods=['POST','GET'])
def Preprocessing():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('Feature_selection'))
    return render_template('Preprocessing.html')

# The function gives us the feature_selection page of the Webapp.
# Call the template preprecssing.html and show feature_selection page.
# If we click on the "next" button, We will jump to Models page
@app.route('/Feature_selection/', methods=['POST','GET'])
def Feature_selection():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('Models'))
    return render_template('Feature_selection.html')

# The function gives us the models page of the Webapp.
# Call the template models.html and show models page.
# If we click on the "next" button, We will jump to diagnosis page
@app.route('/Models/', methods=['POST', 'GET'])
def Models():
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('Diagnosis'))
    return render_template('Models.html')

# The function gives us the diagnosis page of the Webapp.
# Call the template diagnosis.html and retrieve the data we inputted and store
# them to g.text1,g.text2,g.text3 and g.text4, they are area_worst,area_mean,
# concave.points_mean and concave.points_worst. then we retrieve the type of the
# models we selected and retrieve the models which has been trained and saved 
# from models.ipynb, finally predict the prediction of the breast cancer is 
# benign or malignant.
@app.route('/Diagnosis/', methods=['POST', 'GET'])
def Diagnosis():
    if request.method == 'GET':
        return render_template('Diagnosis.html')
    else:
        try:
            # retrieve the input data
            g.text1 = float(request.form['text1'])  #area_worst
            g.text2 = float(request.form['text2'])  #area_mean
            g.text3 = float(request.form['text3'])  #concave.points_mean
            g.text4 = float(request.form['text4'])  #concave.points_worst
            #standarize the input data
            X_value = [g.text1,g.text2,g.text3,g.text4]
            df3 = pd.read_csv(".\\files\\standarize.csv")
            c=["mean","std"]
            df4 = df3.loc[:,c]
            X_value= (X_value-df4["mean"])/df4["std"]
            X_value=np.array(X_value).tolist()
            X_value = [X_value]
            g.m = request.form['model']
            if g.m == "LR":    #if you select Logistic Regression
                g.model = joblib.load(".\\models\\LR.m") 
            elif g.m == "DT":  #if you select Decision Tree
                g.model = joblib.load(".\\models\\DT.m")
            elif g.m == "MLP": #if you select MultilayerPerceptron
                g.model = joblib.load(".\\models\\MLP.m")
            elif g.m == "RF":  #if you select Random Forest
                g.model = joblib.load(".\\models\\RF.m")
            elif g.m == "SVM":  #if you select Support Vector Machine
                g.model = joblib.load(".\\models\\SVM.m")
            else:               #if you select Tensor Flow
                g.model = tf.keras.models.load_model(".\\models\\TF")
            #predict the result
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
            return render_template('Diagnosis.html', result1 = r1,
                                    result2 = r2)
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
  <img src="https://raw.githubusercontent.com/xinyudong1129/xinyudong1129.
  github.io/master/images/analyse_scaled_ra.jpg" width="864px" height="360px">
  <p>  The comparison diagram of TNE before and after standarization</p>
  <img src="https://raw.githubusercontent.com/xinyudong1129/xinyudong1129.
  github.io/master/images/TSNE_TSNE_scaled.png" width="702px" height="280px">
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
    <li>cols1 = ['perimeter_worst','perimeter_mean','area_worst','area_mean',
        'concave.points_mean','concave.points_worst','radius_worst',
        'radius_mean','concavity_worst','concavity_mean']</li> 
    <li>cols2 = ['radius_mean','perimeter_mean','area_mean',
        'concave.points_mean','concavity_mean']</li>
    <li>cols3 = ['perimeter_worst','perimeter_mean','area_worst','area_mean',
        'concave.points_mean','concave.points_worst']</li>
    <li>cols4 = ['perimeter_worst','perimeter_mean','area_worst','area_mean']</li>
    <li>cols5 = ['area_worst','area_mean','concave.points_mean','concave.points_worst']</li>
    <li>cols6 = ['area_mean','concave.points_mean']</li>
  </ul>
<div align = center>
  <img src="https://raw.githubusercontent.com/xinyudong1129/xinyudong1129.
        github.io/master/images/models1.png" width="826px" height="660px">
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

This is the diagnosis page.
```html
{%raw%}
{% extends 'base.html' %}
{% block header %}
  <h1>{% block title %}Diagnosis{% endblock %}</h1>
{% endblock %}
{% block content %}
  <form method="post" enctype="multipart/form-data">
    <h2> Please input data </h2>
    <label for="text1">area_worst</label>
    <input type="text1" name="text1" id="text1">
    <br>
    <label for="text2">area_mean</label>
    <input type="text2" name="text2" id="text2">
    <br>
    <label for="text3">concave.points_mean</label>
    <input type="text3" name="text3" id="text3">
    <br>
    <label for="text4">concave.points_worst</label>
    <input type="text4" name="text4" id="text4">
    <br>
    <h2> Please select a model </h2>
    <input type="radio" name="model" id="LR" value="LR">
    <label for="Logistic Regression">Logistic Regression</label>
    <br>
    <input type="radio" name="model" id="DT" value="DT">
    <label for="Decision Tree">Decision Tree</label>
    <br>
    <input type="radio" name="model" id="MLP" value="MLP">
    <label for="Multilayer Perceptron">Multilayer Perceptron</label>
    <br>
    <input type="radio" name="model" id="RF" value="RF">
    <label for="Random Forest">Random Forest</label>
    <br>
    <input type="radio" name="model" id="SVM" value="SVM">
    <label for="Support Vector Machine">Support Vector Machine</label>
    <br>
    <input type="radio" name="model" id="TF" value="TF">
    <label for="Tensor Flow">Tensor Flow</label>
    <br>
    <input type="submit" value="Diagnosis">
  </form>
  {% if result1 %}
    <br>
    The diagnosis of the breast cancer is malignant.
    <br>
  {% endif %}
  {% if result2 %}
    <br>
    The diagnosis of the breast cancer is benign.
    <br>
  {% endif %}
  {% if error %}
    <br>
    Oh no, we couldn't use that file!  Please upload an 8x8 numpy array as a text file.
  {% endif %}
{% endblock %}
{%endraw%}
```

![diagnosis1.jpg]({{ site.baseurl }}/images/diagnosis1.png)

![diagnosis2.jpg]({{ site.baseurl }}/images/diagnosis2.png)


> This is all about our project, if you want to get more information of our project and see how it works in detail. Please visit our Github Repository. Here is the link [https://github.com/panxinming/PIC16B-Proposal](https://github.com/panxinming/PIC16B-Proposal)
>
> ***Thank You!***