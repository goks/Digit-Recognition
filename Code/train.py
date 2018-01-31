import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

a = pd.read_csv("../Data/train.csv")
b = pd.read_csv("../Data/test.csv")

print (a.shape)
train_Y = pd.DataFrame()
train_Y["label"] = a["label"]

# preprocessing
train_X = a.iloc[:,1:].values.astype('float')
test_X = b.values.astype('float')
max = np.max(train_X)
train_X /= max
test_X /= max

mean = np.std(train_X)
train_X -= mean
test_X -= mean
# print(train_X)

# to binary matrix
train_Y = np_utils.to_categorical(train_Y["label"]).astype('int')


train_dim = train_X.shape[1]
no_classes = train_Y.shape[1]

print(train_X)
print(train_Y)
model = Sequential()

model.add(Dense(units=128, input_dim=train_dim))
model.add(Activation('relu'))
model.add(Dense(no_classes))
model.add(Activation("softmax"))

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
fit = model.fit(train_X, train_Y, epochs=10, batch_size=16, validation_split=0.1,verbose=2 )

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



