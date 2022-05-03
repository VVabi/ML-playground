from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras,lite
from tensorflow.keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
#matplotlib inline

def get_perceptron_mnist_model(input_size):
    inputs  = keras.Input(shape=(input_size,))
    dense = layers.Dense(80, activation="relu", input_dim=input_size)
    outputs = dense(inputs)
    dense2 = layers.Dense(40, activation="relu")
    outputs = dense2(outputs)
    softmax = layers.Dense(10, activation="softmax")
    outputs = softmax(outputs)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist")
    return model

def get_cnn_mnist_model(input_size):
    inputs  = keras.Input(shape=(input_size,))
    reshaped_input = layers.Reshape((28, 28, 1), input_shape=(input_size,))(inputs)
    x = layers.Conv2D(16, 5, strides=(2, 2))(reshaped_input)
    x = layers.Conv2D(32, 5, strides=(2, 2))(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    dense = layers.Dense(80, activation="relu", input_dim=input_size)
    outputs = dense(x)
    dense2 = layers.Dense(40, activation="relu")
    outputs = dense2(outputs)
    softmax = layers.Dense(10, activation="softmax")
    outputs = softmax(outputs)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_cnn")
    return model

mnist = fetch_openml('mnist_784', version=1, cache=True)

X = mnist.data
#Y = np.zeros([70000, 10])

#for ind in range(70000):
#    Y[ind, int(mnist.target[ind])] = 1

Y = np.zeros(mnist.target.shape)

for ind in range(mnist.target.shape[0]):
    Y[ind] = float(mnist.target[ind])

indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices)
X = tf.gather(X, shuffled_indices)
Y = tf.gather(Y, shuffled_indices)


X_train, X_test = X[:60000], X[60000:]

Y_train, Y_test = Y[:60000], Y[60000:]

model = get_perceptron_mnist_model(mnist.data.shape[1])

model.summary()

model.compile(
    loss = losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

history = model.fit(X_train, Y_train, validation_split = 0.05, batch_size=32, epochs=20)

acc = model.evaluate(X_test, Y_test)

print(acc)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

fig, axs = plt.subplots(10, 2,figsize=(10,10))
fig.tight_layout()
preds = model.predict(X_test)
cnt = 0
for ind in range(preds.shape[0]):
    if np.argmax(preds[ind]) != Y_test[ind] and cnt < 10:
        
        image = np.reshape(X_test[ind], (28, 28))
        
        axs[cnt, 0].imshow(image, cmap='gray')
        axs[cnt, 0].axis('off')
        current_pred = preds[ind]
        axs[cnt, 1].set_ylim([0, 1])
        axs[cnt, 1].bar(range(current_pred.shape[0]), current_pred, width=0.5)
        cnt = cnt+1
    
plt.show()

fig, axs = plt.subplots(10, 2,figsize=(10,10))
fig.tight_layout()

for ind in range(10):
        image = np.reshape(X_test[ind], (28, 28))
        axs[ind, 0].imshow(image, cmap='gray')
        axs[ind, 0].axis('off')
        current_pred = preds[ind]
        axs[ind, 1].set_ylim([0, 1])
        axs[ind, 1].bar(range(current_pred.shape[0]), current_pred)
    
plt.show()