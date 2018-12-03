import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#dataframe = read_csv("international-airline-passengers.csv", usecols=[1],engine='python', skipfooter=3)
#print(dataframe)
#dataset = dataframe.values
#print(dataset)
#dataset = dataset.astype('float32')

dataset = numpy.loadtxt("groundtruth_rect.txt")
print(dataset)
#plt.plot(dataset)
#plt.show()

def create_dataset(dataset, look_back=1):
    dataX, dataY =[], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i+look_back, :])
    return numpy.array(dataX), numpy.array(dataY)


numpy.random.seed(7)

#对于检测框的变化不做归一化处理
#scaler = MinMaxScaler(feature_range=(0,1))
#dataset = scaler.fit_transform(dataset)


train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
print(train_size, test_size)
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset),:]

look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(trainY.shape)
print('--------')
#print(trainX)
#print(trainY)
#reshape input to be [samples, time steps, features]
#print(trainX.shape[0], trainX.shape[1])
#print(trainX)
#print(trainX.shape)
trainX = numpy.reshape(trainX, (trainX.shape[0],  trainX.shape[1], trainX.shape[2]))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))
#print(trainX)
print(trainX.shape)

#create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 4), return_sequences = False))
#model.add(LSTM(4, return_sequences=False))
model.add(Dense(4))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

model.save("model.h5")
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#invert precictions
# trainPredict = scaler.inverse_transform(trainPredict)
# print(trainY.shape)
# trainY = scaler.inverse_transform(trainY)
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform(testY)

trainScore = math.sqrt(mean_squared_error(trainY[:, :], trainPredict[:, :]))
#print(trainY)
#print(trainY[0])
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = numpy.empty_like(dataset[:,0:2])
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict[:,0:2]

testPredictPlot = numpy.empty_like(dataset[:,0:2])
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict[:,0:2]

plt.plot(dataset[:,0:2])
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
