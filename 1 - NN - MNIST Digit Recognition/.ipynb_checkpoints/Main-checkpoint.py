from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from sklearn.model_selection import train_test_split


def tfds_to_numpy(dataset):
    """ Convert tf.data.Dataset to numpy arrays
    Args: 
        dataset (tf.data.Dataset) : MNIST dataset
    Returns: 
        (np.array) : Data
        (np.array) : Labels  
    """
    images = []
    labels = []
    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

def build_model(L1_nodes,L2_nodes,learning_rate):
    """ Build neural-network model
    Args: 
        L1_nodes (int) : Number of nodes in first layer
        L2_nodes (int) : Number of nodes in second layer
        learning_rate (float) : Neural network learning rate
    Returns: 
        model (Keras model) : Neural network model
     """
    tf.random.set_seed(1234)
    model = Sequential(
        [
            keras.Input(shape=(784,)),
            Dense(L1_nodes, activation="relu", name="L1"),
            Dense(L2_nodes, activation="relu", name="L2"),
            Dense(10, activation="linear", name="L3")
        ], name="model"
    )
    
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    )
    return model

def get_best_result_2d(j_cv):
    """ Return index of lowest loss in the matrix
    Args:
        j_cv (array(M,N)) : Loss matrix 
    Returns:
        index (array(M,N)) : Position in the matrix with lowest loss 
    """
    index = [0,0]
    auxVal = 9000
    
    for i in range(0,j_cv.shape[0]):
        for l in range(0,j_cv.shape[1]):
            if j_cv[i][l] < auxVal:
                index = [i,l]
                auxVal = j_cv[i][l]
    return np.array(index)

def get_best_result(j_cv):
    """ Return index of lowest loss in the array
    Args:
        j_cv (array(M)) : Loss array 
    Returns:
        index (int) : Position in the array with lowest loss 
    """
    index = 0
    auxVal = 9000
    
    for i in range(0,j_cv.shape[0]):
            if j_cv[i] < auxVal:
                index = i
                auxVal = j_cv[i]
    return index
    
# Load MNIST Dataset
ds = tfds.load('mnist', split='train', as_supervised=True)
ds_test = tfds.load('mnist', split='test', as_supervised=True)

# Convert trainning data
train_images, Y_train = tfds_to_numpy(ds)

# Convert testing data
test_images, Y_test = tfds_to_numpy(ds_test)

# Show an image from the trainning set
plt.imshow(train_images[0], cmap='gray')
plt.title(f'Label: {Y_train[0]}')
plt.show()

# N - Number of samples
# Reshape N*28*28 to N*784
X_train = np.float_(np.reshape(train_images,(train_images.shape[0],train_images.shape[1]*train_images.shape[2])))
X_test = np.float_(np.reshape(test_images,(test_images.shape[0],test_images.shape[1]*test_images.shape[2])))

# Normalize data
X_train /= 255.0
X_test /= 255.0

# Split test set into cross-validation set and test set
X_CV, X_test, Y_CV, Y_test = train_test_split(X_test, Y_test, test_size = 0.5, random_state = 0)

# Hyperparameters tunning
# Tunning L1 and L2 number of nodes from 14 to 17 and 12 to 13 respectively (there was already a prior tunning and selection to reach these numbers)
learning_rate=0.001
from_i=14
to_i=17
from_l=12
to_l=13
j_cv = np.zeros([to_i-from_i,to_l-from_l])
j_train = np.zeros([to_i-from_i,to_l-from_l])
for i in range(from_i,to_i):
    for l in range(from_l,to_l):
        model = build_model(i,l,learning_rate)
        history = model.fit(X_train,Y_train,epochs=40)
        j_train[i-from_i][l-from_l] = history.history['loss'][-1]
        j_cv[i-from_i][l-from_l] = model.evaluate(X_CV,Y_CV)

index = get_best_result_2d(j_cv)
print(np.array(index)+np.array([from_i,from_l]),j_cv[index[0]][index[1]])
print(j_train)
print(j_cv)
# Best result: L1-15 Nodes L2- 12 Nodes

# Determine number of epochs that produces the lowest loss in cross-validation set (there was already a prior tunning and selection to reach these numbers)
learning_rate=0.001
model = build_model(15,12,learning_rate)
j_cv2 = np.zeros(3)
j_train2 = np.zeros(3)
ind=0
for i in range(6,9):
    history = model.fit(X_train,Y_train,epochs=i)
    j_train2[ind] = history.history['loss'][-1]
    j_cv2[ind] = model.evaluate(X_CV,Y_CV)
    ind+=1
print(j_cv2,j_train2)

index = get_best_result(j_cv2)
print(index)
# Best result: 7 epochs

# Trainning a model with the best parameters and printing cross-validation and test loss
learning_rate=0.001
model = build_model(15,12,learning_rate)
model.fit(X_train,Y_train,epochs=7)
j_cv_result = model.evaluate(X_CV,Y_CV)
j_test_result = model.evaluate(X_test,Y_test)
print(j_cv_result,j_test_result)


# Testing model 

test_input_ex = X_test[210]
test_label_ex = Y_test[210]

plt.imshow(test_input_ex.reshape([28,28]),cmap='gray')
plt.title(test_label_ex)
plt.show()
np.argmax(tf.nn.softmax(model.predict(test_input_ex.reshape([1,784]))))

pred = np.reshape(X_test[65],[1,-1])
print(np.argmax(tf.nn.softmax(model.predict(pred))))
print(Y_CV[65])

plt.imshow(np.reshape(X_CV[65], [28,28]),cmap="gray")
plt.title("Number")
plt.show()