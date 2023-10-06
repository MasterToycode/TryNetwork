from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.datasets import mnist
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
tensorboard = TensorBoard(log_dir='logs')
# 加载MNIST数据集
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

# 归一化输入数据
x_train = x_train / 255.0
x_valid = x_valid / 255.0

# 将图像数据展平为一维向量
x_train = x_train.reshape(-1, 784)
x_valid = x_valid.reshape(-1, 784)

# 将标签数据进行One-Hot编码
y_train = to_categorical(y_train, num_classes=10)
y_valid = to_categorical(y_valid, num_classes=10)

# 创建模型
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(BatchNormalization())


model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 训练模型
model.fit(
    x_train, y_train,
    batch_size=128, epochs=30, verbose=1,
    validation_data=(x_valid, y_valid),
   # callbacks=[tensorboard]
)
