
"""
Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/kmkarakaya/ML_tutorials/blob/master/Conv1d_Predict_house_prices.ipynb

# Include Libraries and Auxiliary Functions
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mae']),
           label='Train')
  plt.plot(history.epoch, np.array(history.history['val_mae']),
           label = 'Val')
  plt.legend()
  plt.ylim([0,max(history.history['val_mae'])])
  plt.show()

def plot_prediction(test_labels, test_predictions):
  plt.figure()
  plt.scatter(test_labels, test_predictions)
  plt.xlabel('True Values [1000$]')
  plt.ylabel('Predictions [1000$]')
  plt.axis('equal')
  plt.xlim(plt.xlim())
  plt.ylim(plt.ylim())
  _ = plt.plot([-100, 100],[-100,100])

  plt.figure()
  error = test_predictions - test_labels
  plt.hist(error, bins = 50)
  plt.xlabel("Prediction Error [1000$]")
  _ = plt.ylabel("Count")
  plt.show()

"""# Notice
***As a base model***, I will use the **TensorFlow official example** for ***MLP model*** and compare its performance with **my Conv1D model**. Thus, we will be able to observe the relative success of **Conv1D model** with respect to **a professional sample model**.

You can access the original notebook ["Predict house prices: regression" with Multi-layer Perceptron here.](https://colab.research.google.com/github/MarkDaoust/models/blob/add-regression-plots/samples/core/tutorials/keras/basic_regression.ipynb)

If you run this notebook,  you would generate mean absolute error values different than the reported ones here due to stochastic nature of ANNs.

# What is regression?

In a *regression* problem, we aim to predict the output of a continuous value, like a price or a probability. Contrast this with a *classification* problem, where we aim to predict a discrete label (for example, where a picture contains an apple or an orange).

This notebook builds two different models to predict the median price of homes in a Boston suburb during the mid-1970s. To do this, I'll provide the models with some data points about the suburb, such as the crime rate and the local property tax rate.

## The Boston Housing Prices dataset

This [dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) is accessible directly in TensorFlow. Download and shuffle the training set:
"""

boston_housing = tf.keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

"""### Examples and features

This dataset has 506 total examples which are split between **404** training examples and **102** test examples:
"""

print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features

"""The dataset contains 13 different features:

1.   Per capita crime rate.
2.   The proportion of residential land zoned for lots over 25,000 square feet.
3.   The proportion of non-retail business acres per town.
4.   Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
5.   Nitric oxides concentration (parts per 10 million).
6.   The average number of rooms per dwelling.
7.   The proportion of owner-occupied units built before 1940.
8.   Weighted distances to five Boston employment centers.
9.   Index of accessibility to radial highways.
10.  Full-value property-tax rate per $10,000.
11.  Pupil-teacher ratio by town.
12.  1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
13.  Percentage lower status of the population.

Each one of these input data features is stored using a different scale. Some features are represented by a proportion between 0 and 1, other features are ranges between 1 and 12, some are ranges between 0 and 100, and so on. This is often the case with real-world data, and understanding how to explore and clean such data is an important skill to develop.

"""

print(train_data[0])  # Display sample features, notice the different scales

"""### Labels

The labels are the house prices in thousands of dollars. (You may notice the mid-1970s prices.)
"""

print(train_labels[0:10])  # Display first 10 entries

"""## Normalize features

It's recommended to normalize features that use different scales and ranges. For each feature, subtract the mean of the feature and divide by the standard deviation:
"""

# Test data is *not* used when calculating the mean and std.
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First training sample, normalized

"""Although the model *might* converge without feature normalization, it makes training more difficult, and it makes the resulting model more dependant on the choice of units used in the input.

## Create the MLP model

Let's build an MLP model. Here, we'll use a `Sequential` model with two densely connected hidden layers, and an output layer that returns a single, continuous value. The model building steps are wrapped in a function, `build_model`, since we'll create a second model, later on.
"""

def build_model():
  model = keras.Sequential([
    keras.layers.Input(shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ], name="MLP_model")

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

"""## Train the model

The model is trained for 500 epochs, and record the training and validation accuracy in the `history` object.
"""

EPOCHS = 500
# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=1)

"""Visualize the model's training progress using the stats stored in the `history` object. We want to use this data to determine how long to train *before* the model stops making progress."""

plot_history(history)

"""The graph shows the average error **in Validation set** is about $2,600 dollars. Is this good?

Well, \$2,600 is not an insignificant amount when some of the labels are only $15,000.

Let's see how did the model performs on the **Test set**:
"""

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

"""## Predict

Finally, predict some housing prices using data in the testing set:
"""

test_predictions = model.predict(test_data).flatten()
plot_prediction(test_labels, test_predictions)

"""## Observations

So far, we implemented a MLP model to handle the "**Boston House Prices**" regression problem.

Mean Absolute Error in Validation is around **\$2,600** whereas in Testing, it is about **$2900**

## Create the Conv1D model

Let's build an Conv1D model. Here, we'll use a `Sequential` model with 3 Conv1D layers, one MaxPooling1D layer, and an output layer that returns a single, continuous value. The model building steps are wrapped in a function, `build_model` as we did above.

## Reshape Data sets
As you might remember, Conv1D layer expects input shape in 3D as

  `[batch_size, time_steps, input_dimension]`

However, current data is in the shape of

`[batch_size, features]`

See below:
"""

print(train_data.shape)
print(train_data[0].shape)
print(train_data[0])

"""That is, in the current data set each sample has 13 features and no timesteps!

**Basically, we convert features to timesteps**

To convert 2D of input data into a 3D input, we simply **reshape** as follows:
"""

sample_size = train_data.shape[0] # number of samples in train set
time_steps  = train_data.shape[1] # number of features in train set
input_dimension = 1               # each feature is represented by 1 number

train_data_reshaped = train_data.reshape(sample_size,time_steps,input_dimension)
print("After reshape train data set shape:\n", train_data_reshaped.shape)
print("1 Sample shape:\n",train_data_reshaped[0].shape)
print("An example sample:\n", train_data_reshaped[0])

"""After conversion, we have a train data set whose shape is

  `[batch_size, time_steps, input_dimension]` ---> `[404, 13, 1]`

That is, each sample has **13 time steps with 1 input dimension**. You can also think as `each sample has 13 rows 1 column`!

##Reminder
* `Conv1D(filters=1, kernel_size=7, activation='relu')`

<img src="https://github.com/kmkarakaya/ML_tutorials/blob/master/images/conv1d.gif?raw=true" width="500">

We need to **reshape** the Test data as well:
"""

test_data_reshaped = test_data.reshape(test_data.shape[0],test_data.shape[1],1)

"""Now, we can create Conv1D model as below."""

def build_conv1D_model():

  n_timesteps = train_data_reshaped.shape[1] #13
  n_features  = train_data_reshaped.shape[2] #1
  model = keras.Sequential(name="model_conv1D")
  model.add(keras.layers.Input(shape=(n_timesteps,n_features)))
  model.add(keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', name="Conv1D_1"))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', name="Conv1D_2"))

  model.add(keras.layers.Conv1D(filters=16, kernel_size=2, activation='relu', name="Conv1D_3"))

  model.add(keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(32, activation='relu', name="Dense_1"))
  model.add(keras.layers.Dense(n_features, name="Dense_2"))


  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])
  return model

model_conv1D = build_conv1D_model()
model_conv1D.summary()

"""## Train the model

The model is trained for 500 epochs, and record the training and validation accuracy in the `history` object.
"""

# Store training stats
history = model_conv1D.fit(train_data_reshaped, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=1)

"""Visualize the model's training progress using the stats stored in the `history` object. We want to use this data to determine how long to train *before* the model stops making progress."""

plot_history(history)

"""The graph shows the average error **in Validation set** is about $2,400 dollars. Is this good?

Well, \$2,400 is not an insignificant amount when some of the labels are only $15,000.

Let's see how did the model performs on the **Test set**:
"""

[loss, mae] = model_conv1D.evaluate(test_data_reshaped, test_labels, verbose=0)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

"""## Predict

Finally, predict some housing prices using data in the testing set:
"""

test_predictions = model_conv1D.predict(test_data_reshaped).flatten()
plot_prediction(test_labels, test_predictions)

model_conv1D.save("models/model_conv1D.keras")

"""## Observations

So far, we implemented an ***MLP model*** and a ***Conv1D model*** to handle the "**Boston House Prices**" regression problem.

MLP model generates Mean Absolute Error:
* in Validation is around **\$2,600** whereas in Testing, it is about **$2900**

Conv1D model generates Mean Absolute Error:
* in Validation is around **\$2,400** whereas in Testing, it is about **$2,700**

Thus, **Conv1D** model is a **competitive** approach considering **MLP** model in the regression problem at hand.

To use a Conv1D model, you need to **reshape** the input as

`[batch_size, time_steps, input_dimension]`

**I hope this tutorial helps you to use Conv1D layer successfuly!**



"""