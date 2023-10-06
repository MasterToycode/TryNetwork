from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.utils import to_categorical
from keras.datasets import boston_housing
import numpy as np
(x_train,y_train),(x_vaild,y_vaild)=boston_housing.load_data()

model=Sequential()

model.add(Dense(32,input_dim=13,activation='relu'))
#input_shape=(13,)也是是正确的！
model.add(BatchNormalization())

model.add(Dense(16,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(1,activation='linear'))

model.compile(loss='mean_squared_error',optimizer='adam')
#这是一个回归问题，我们使用relu神经元没有意义，我们不需要概率，需要回归预测的真实数值。所以也不使用交叉熵损失函数，而是使用回归问题的特定的损失函数！
model.fit(
    x_train,y_train,
    batch_size=8,epochs=32,verbose=1,
    validation_data=(x_vaild,y_vaild)
)

for i in range(len(x_vaild)):
    prediction = model.predict(np.reshape(x_vaild[i], [1, 13]))[0][0]
    true_value = y_vaild[i]
    difference = true_value - prediction
    print(f"样本{i + 1}的预测房价: {prediction}, 真实房价: {true_value}, 差异: {difference}")

