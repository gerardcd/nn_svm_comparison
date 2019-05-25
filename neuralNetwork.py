from matplotlib import pyplot as plt
import math

from pandas import DataFrame
from keras.layers import Dense, Dropout
from keras.models import Sequential

N = 1024

hidden_neurons = 32
batch_size = int(math.log(N) / math.log(2))
num_classes = 2
epochs = 32
hidden_activation = 'relu'
output_activation = 'softmax'
optimizer = 'RMSprop'
loss = 'categorical_crossentropy'
dropout_rate = 0.1


def encode(value):
    value = int(value)

    return [1,0] if value == 1 else [0,1]


def dataSetFromFile(fileName):
    xData = []
    yData = []
    with open(fileName) as file:
        for row in file:
            values = row.split()

            xData.append(values[:-1])

            yData.append(
                encode(values[-1])
            )

    return (DataFrame(xData), DataFrame(yData))


(xTrain, yTrain) = dataSetFromFile('shuttle.trn')
(xTest, yTest) = dataSetFromFile('shuttle.tst')

xTrain = xTrain[:N]
yTrain = yTrain[:N]

model = Sequential()
model.add(Dense(hidden_neurons, activation=hidden_activation, input_shape=(9,)))
model.add(Dropout(dropout_rate))

model.add(Dense(num_classes, activation=output_activation))

model.summary()

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit(xTrain, yTrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(xTest, yTest))

score = model.evaluate(xTest, yTest, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()