import tensorflow as tf
import matplotlib.pyplot as plt

#Loading dataset
(trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#One Hot Encoding Labels from Train and Test Dataset
trainY_one_hot = tf.keras.utils.to_categorical(trainY)
testY_one_hot = tf.keras.utils.to_categorical(testY)

#Initializing MobileNet as Base Model for Transfer Learning
base_model = tf.keras.applications.MobileNet(include_top=False, weights='imagenet', input_shape=(32,32,3), classes=trainY.shape[1])

#Adding layers to base model of MobileNet
model = tf.keras.Sequential()
#Creating base layer of VGG19
model.add(base_model)
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
#Adding the Dense Layers and Dropout
model.add(tf.keras.layers.Dense(512, activation=('relu'))) 
model.add(tf.keras.layers.Dense(256, activation=('relu'))) 
model.add(tf.keras.layers.Dropout(.3))
model.add(tf.keras.layers.Dense(128, activation=('relu')))
model.add(tf.keras.layers.Dropout(.2))
model.add(tf.keras.layers.Dense(10, activation=('softmax')))

#Visualizing Model Summary
print(model.summary())
#Compiling Model using SGD
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#Training Model
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
hist = model.fit(trainX, trainY_one_hot, batch_size = 100, epochs = 20, validation_split = 0.1, callbacks=[es])

#Visualizing Model Accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc = 'upper left')

#Testing accuracy of trained model
test_loss, test_acc = model.evaluate(testX, testY_one_hot)

print("test acc: ", test_acc)

# Save the entire model as a SavedModel
model.save('./model')