---
layout: post
title: Blog Post 6
---


## Fake News Classification
In this Blog Post, we will develop and assess a fake news classifier using Tensorflow.
Our data for this assignment comes from the article

>Ahmed H, Traore I, Saad S. (2017) “Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. In: Traore I., Woungang I., Awad A. (eds) Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments. ISDDC 2017. Lecture Notes in Computer Science, vol 10618. Springer, Cham (pp. 127-138).


## (A). Acquire Training Data

We are going to load the data we need.
```python
import pandas as pd
import numpy as np

train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
df = pd.read_csv(train_url, header=0, index_col=0)
```

## (B). Make a Dataset
Write a function called make_dataset. This function should do two things:
1. Remove stopwords from the article text and title. A stopword is a word that is usually considered to be uninformative, such as “the,” “and,” or “but.”
2. Construct and return a tf.data.Dataset with two inputs and one output. The input should be of the form (title, text), and the output should consist only of the fake column. 

```python
import tensorflow as tf
from nltk.corpus import stopwords
stop = stopwords.words('english')


def make_dataset(df):
    df['title'] = df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    data = tf.data.Dataset.from_tensor_slices(
        (
            {
                "title" : df[["title"]], 
                "text" : df[["text"]]
            }, 
            {
                "fake" : df[["fake"]]
            }
        )
    )

    # We wish to shuffle the dataset
    my_data_set = data.shuffle(buffer_size = len(data))

    # Using the batch function we group entries
    my_data_set.batch(100)

    return my_data_set
    
data = make_dataset(df)
```

After we’ve constructed our primary Dataset, split of 20% of it to use for validation.
```python
train_size = int(0.8*len(data))
val_size   = int(0.2*len(data))

train = data.take(train_size)
val   = data.skip(train_size).take(val_size)

len(train), len(val)
```

Standardization refers to the act of taking a some text that's "messy" in some way and making it less messy. Common standardizations include:
- Removing capitals.
- Removing punctuation.
- Removing HTML elements or other non-semantic content.

Vectorization refers to the process of representing text as a vector (array, tensor). 

So, then we do the **Standardization and Vectorization**

```python
import string
import re

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import Input, layers, Model, utils, losses

size_vocabulary = 2000

def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation 

vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 

vectorize_layer.adapt(train.map(lambda x, y: x["text"]))
embedding = layers.Embedding(size_vocabulary, 3, name = "embedding")
```

## (C). Create Models

To address this question, create three (3) TensorFlow models.

- In the first model, you should use only the article title as an input.
- In the second model, you should use only the article text as an input.
- In the third model, you should use both the article title and the article text as input.

### (1). First Model: Only the article title
```python
title_input = Input(shape = (1,), name = "title", dtype = "string")

# Text Embedding
title_features = vectorize_layer(title_input)
title_features = embedding(title_features)

# Normal Network Architecture
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(32, activation='relu')(title_features)
output = layers.Dense(1, name = "fake")(title_features)

title_model = Model(
   inputs = [title_input],
   outputs = output
)
title_model.compile(optimizer = "adam",
                   loss = losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy']
)

history = title_model.fit(train, 
                         validation_data=val,
                         epochs = 10)
```
```
Epoch 1/10
17959/17959 [==============================] - 20s 1ms/step - loss: 0.3816 - accuracy: 0.7808 - val_loss: 0.1422 - val_accuracy: 0.9577
Epoch 2/10
17959/17959 [==============================] - 20s 1ms/step - loss: 0.1630 - accuracy: 0.9330 - val_loss: 0.1429 - val_accuracy: 0.9595
Epoch 3/10
17959/17959 [==============================] - 20s 1ms/step - loss: 0.1362 - accuracy: 0.9471 - val_loss: 0.0801 - val_accuracy: 0.9726
Epoch 4/10
17959/17959 [==============================] - 20s 1ms/step - loss: 0.1285 - accuracy: 0.9501 - val_loss: 0.0883 - val_accuracy: 0.9719
Epoch 5/10
17959/17959 [==============================] - 20s 1ms/step - loss: 0.1161 - accuracy: 0.9546 - val_loss: 0.0716 - val_accuracy: 0.9755
Epoch 6/10
17959/17959 [==============================] - 20s 1ms/step - loss: 0.1133 - accuracy: 0.9542 - val_loss: 0.0647 - val_accuracy: 0.9800
Epoch 7/10
17959/17959 [==============================] - 20s 1ms/step - loss: 0.1104 - accuracy: 0.9555 - val_loss: 0.0733 - val_accuracy: 0.9706
Epoch 8/10
17959/17959 [==============================] - 20s 1ms/step - loss: 0.1096 - accuracy: 0.9557 - val_loss: 0.1757 - val_accuracy: 0.9443
Epoch 9/10
17959/17959 [==============================] - 20s 1ms/step - loss: 0.1086 - accuracy: 0.9567 - val_loss: 0.1143 - val_accuracy: 0.9485
Epoch 10/10
17959/17959 [==============================] - 20s 1ms/step - loss: 0.1058 - accuracy: 0.9573 - val_loss: 0.1108 - val_accuracy: 0.9530
```

Then, we plot the history of the accuracy on both the training and validation sets.
```python
from matplotlib import pyplot as plt
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```

![fake1.jpg]({{ site.baseurl }}/images/fake1.png)


### (2). Second Model: Only the article text

```python
text_input = Input(shape = (1,), name = "text", dtype = "string")

# Text Embedding
text_features = vectorize_layer(text_input)
text_features = embedding(text_features)

# Normal Network Architecture
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(32, activation='relu')(text_features)
output = layers.Dense(1, name = "fake")(text_features)

text_model = Model(
   inputs = [text_input],
   outputs = output
)
text_model.compile(optimizer = "adam",
                   loss = losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy']
)

history = text_model.fit(train, 
                         validation_data=val,
                         epochs = 10)
```
```
Epoch 1/10
17959/17959 [==============================] - 23s 1ms/step - loss: 0.2776 - accuracy: 0.8775 - val_loss: 0.1848 - val_accuracy: 0.9479
Epoch 2/10
17959/17959 [==============================] - 23s 1ms/step - loss: 0.1758 - accuracy: 0.9388 - val_loss: 0.1306 - val_accuracy: 0.9644
Epoch 3/10
17959/17959 [==============================] - 22s 1ms/step - loss: 0.1415 - accuracy: 0.9536 - val_loss: 0.1082 - val_accuracy: 0.9521
Epoch 4/10
17959/17959 [==============================] - 22s 1ms/step - loss: 0.1227 - accuracy: 0.9616 - val_loss: 0.1160 - val_accuracy: 0.9530
Epoch 5/10
17959/17959 [==============================] - 23s 1ms/step - loss: 0.1114 - accuracy: 0.9634 - val_loss: 0.1004 - val_accuracy: 0.9737
Epoch 6/10
17959/17959 [==============================] - 22s 1ms/step - loss: 0.0994 - accuracy: 0.9681 - val_loss: 0.0873 - val_accuracy: 0.9762
Epoch 7/10
17959/17959 [==============================] - 22s 1ms/step - loss: 0.0974 - accuracy: 0.9693 - val_loss: 0.0692 - val_accuracy: 0.9806
Epoch 8/10
17959/17959 [==============================] - 23s 1ms/step - loss: 0.0897 - accuracy: 0.9712 - val_loss: 0.0635 - val_accuracy: 0.9824
Epoch 9/10
17959/17959 [==============================] - 23s 1ms/step - loss: 0.0857 - accuracy: 0.9736 - val_loss: 0.0462 - val_accuracy: 0.9871
Epoch 10/10
17959/17959 [==============================] - 23s 1ms/step - loss: 0.0764 - accuracy: 0.9753 - val_loss: 0.0585 - val_accuracy: 0.9877
```

Then, we plot the history of the accuracy on both the training and validation sets.
```python
from matplotlib import pyplot as plt
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```

![fake2.jpg]({{ site.baseurl }}/images/fake2.png)

### (3). Third Model: Both the article title and the article text

```python
combined_features = layers.concatenate([text_features, title_features], axis = 1)
combined_features = layers.Dense(10)(combined_features)
output = layers.Dense(1, name = "fake")(combined_features)

both_model = Model(
   inputs = [text_input, title_input],
   outputs = output
)

both_model.compile(optimizer = "adam",
                   loss = losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy']
)

history = both_model.fit(train, 
                         validation_data=val,
                         epochs = 10)
``` 
```
Epoch 1/10
17959/17959 [==============================] - 25s 1ms/step - loss: 0.0508 - accuracy: 0.9825 - val_loss: 0.0289 - val_accuracy: 0.9869
Epoch 2/10
17959/17959 [==============================] - 25s 1ms/step - loss: 0.0431 - accuracy: 0.9851 - val_loss: 0.0309 - val_accuracy: 0.9931
Epoch 3/10
17959/17959 [==============================] - 25s 1ms/step - loss: 0.0415 - accuracy: 0.9858 - val_loss: 0.0254 - val_accuracy: 0.9918
Epoch 4/10
17959/17959 [==============================] - 25s 1ms/step - loss: 0.0387 - accuracy: 0.9858 - val_loss: 0.0282 - val_accuracy: 0.9900
Epoch 5/10
17959/17959 [==============================] - 25s 1ms/step - loss: 0.0392 - accuracy: 0.9865 - val_loss: 0.0318 - val_accuracy: 0.9913
Epoch 6/10
17959/17959 [==============================] - 25s 1ms/step - loss: 0.0345 - accuracy: 0.9884 - val_loss: 0.0241 - val_accuracy: 0.9922
Epoch 7/10
17959/17959 [==============================] - 25s 1ms/step - loss: 0.0374 - accuracy: 0.9865 - val_loss: 0.0242 - val_accuracy: 0.9889
Epoch 8/10
17959/17959 [==============================] - 25s 1ms/step - loss: 0.0350 - accuracy: 0.9880 - val_loss: 0.0186 - val_accuracy: 0.9915
Epoch 9/10
17959/17959 [==============================] - 25s 1ms/step - loss: 0.0332 - accuracy: 0.9875 - val_loss: 0.0185 - val_accuracy: 0.9922
Epoch 10/10
17959/17959 [==============================] - 26s 1ms/step - loss: 0.0333 - accuracy: 0.9882 - val_loss: 0.0201 - val_accuracy: 0.9924
```

Then, we plot the history of the accuracy on both the training and validation sets.
```python
from matplotlib import pyplot as plt
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.ylim(0.9, 1)
plt.legend()
```

![fake3.jpg]({{ site.baseurl }}/images/fake3.png)


## (D). Model Evaluation
Now we’ll test our model performance on unseen test data. For this part, we plan to focus on our best model, and ignore the other two.

Once we’re satisfied with your best model’s performance on validation data, download the test data here:
```python
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true"
test_df = pd.read_csv(test_url)
test_dataset = make_dataset(test_df)
both_model.evaluate(test_dataset)
```
```
[0.03792494907975197, 0.9861463904380798]
```

It gives us 98.61% accuracy. It's very nice.

## (E). Embedding Visualization
Visualize and comment on the embedding that our model learned. We are able to find some interesting patterns or associations in the words that the model found useful when distinguishing real news from fake news.

```python
weights = both_model.get_layer('embedding').get_weights()[0] # get the weights from the embedding layer
vocab = vectorize_layer.get_vocabulary() # get the vocabulary from our data prep for later
weights
```
```
array([[-4.3807211e-03,  3.3131484e-03,  2.5179882e-02],
       [ 3.0498058e-01,  2.1129368e-01, -5.3251249e-01],
       [-4.7779570e+00, -4.9961200e+00,  4.7236590e+00],
       ...,
       [ 8.3289188e-01,  1.0012127e+00, -1.2732483e+00],
       [ 1.0635720e+00,  1.0107019e+00, -9.6477473e-01],
       [ 4.0210972e+00,  4.0266252e+00, -4.2613201e+00]], dtype=float32)
```

The collection of weights is 3-dimensional. For plotting in 2 dimensions, we have several choices for how to reduce the data to a 2d representation. A very simple and standard approach is our friend, principal component analysis (PCA).
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
weights = pca.fit_transform(weights)
# Now we'll make a data frame from our results:
embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})
embedding_df
```

{% include hhh.html %}


Ready to plot! Note that the embedding appear to be "stretched out" in three directions, with one direction corresponding to each of the three categories (tech, style, science).
```python
import plotly.express as px 
fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = list(np.ones(len(embedding_df))),
                 size_max = 2,
                 hover_name = "word")

fig.show()
```

{% include fake3.html %}

Cool, we made a word embedding! This embedding seems to have learned some reasonable associations.


Whenever we create a machine learning model that might conceivably have impact on the thoughts or actions of human beings, we have a responsibility to understand the limitations and biases of that model. Biases can enter into machine learning models through several routes, including the data used as well as choices made by the modeler along the way.

With these considerations in mind, let's see what kinds of words our model associates with China and Trump.

```python
country = ["china"]
people = ["trump"]

highlight_1 = ["cyber", "oil", "died","alliance"]
highlight_2 = ["guns", "voting", "offensive"]

def gender_mapper(x):
    if x in country:
        return 1
    elif x in people:
        return 4
    elif x in highlight_1:
        return 3
    elif x in highlight_2:
        return 2
    else:
        return 0

embedding_df["highlight"] = embedding_df["word"].apply(gender_mapper)
embedding_df["size"]      = np.array(1.0 + 50*(embedding_df["highlight"] > 0))
```

```python
import plotly.express as px 

fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 color = "highlight",
                 size = list(embedding_df["size"]),
                 size_max = 10,
                 hover_name = "word")

fig.show()
```

{% include fake4.html %}

Our text classification model's word embedding has some bias.

- Words like "Trump" are more closely located to "China".
- Words like "guns", "voting", "offensive" are more closely located to "Trump".

Where did these biases come from?

- Trump mentioned China many times during news conference.
- Trump is not liked by most of United States Citizen. So, many bad news may come through him, Like "Guns", "offensive". However, gun crime is not only the problem when Trump is president, but also other presidents.