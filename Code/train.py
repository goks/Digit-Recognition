import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPool2D, Flatten
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

a = pd.read_csv("../Data/train.csv")
b = pd.read_csv("../Data/test.csv")

# print (a.shape)
train_Y = pd.DataFrame()
train_Y["label"] = a["label"]

# preprocessing
train_X = a.iloc[:,1:].values.astype('float')
test_X = b.values.astype('float')
max = 255
train_X /= 255
test_X /= 255

train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

# mean = np.std(train_X)
# train_X -= mean
# test_X -= mean
# print(train_X)

# to binary matrix
train_Y = np_utils.to_categorical(train_Y["label"]).astype('int')

# print(train_Y.shape)


np.random.seed(2)
random_seed = 2

train_X, train_test_X, train_Y, train_test_Y = train_test_split(train_X, train_Y, test_size=0.1, random_state=random_seed)

# g = plt.imshow(train_X[0][:, :, 0])

print(train_X.shape)
print(train_Y.shape)
model = Sequential()

model.add(Conv2D(32,(5, 5), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32,(5, 5), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3, 3), activation='relu'))
model.add(Conv2D(64,(3, 3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr=0.001), metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
datagen = ImageDataGenerator()
fit = model.fit_generator(datagen.flow(train_X, train_Y, batch_size=96), epochs=1,
                          validation_data=(train_test_X, train_test_Y), steps_per_epoch=train_X.shape[0],
                          callbacks = [learning_rate_reduction])
plt.plot(fit.history['acc'])
plt.plot(fit.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train','test'],loc='upper left')

plt.show()

plt.plot(fit.history['loss'])
plt.plot(fit.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train','test'],loc='upper left')

plt.show()

preds_raw = model.predict(test_X)
predictions = preds_raw.argmax(axis=-1)
pred = pd.DataFrame({'ImageId' : list(range(1,len(predictions)+1)), 'Label': predictions })
pred.to_csv("../Out/predictions.csv", index=False)
print(pred)



