from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D #new!
#Conv2D构建卷积层，MaxPooling2D用于构建池化层
from keras.layers import Flatten #new!
#平铺3位矩阵为一维矩阵，全连接层用于接受和输出，全连接层只接受一维数组！

(x_train,y_train),(x_valid,y_valid)=mnist.load_data()
x_train=x_train.reshape(60000,28,28,1).astype('float32')
x_valid=x_valid.reshape(10000,28,28,1).astype('float32')

x_train=x_train/255.0
x_valid=x_valid/255.0

# 将标签数据进行One-Hot编码
y_train = to_categorical(y_train, num_classes=10)
y_valid = to_categorical(y_valid, num_classes=10)

model=Sequential()

#第一个卷积层
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
#第二个卷积层
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
#全连接层
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
#输出层
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#model.summary()
#虽然池化层，Dropout层,Flatten()层不是人工神经元的组成部分，神经网络并不会认为这个是深度学习中的隐藏层，但是，我们仍然可以添加进我的模型中
# 训练模型
model.fit(
    x_train, y_train,
    batch_size=128, epochs=10, verbose=1,
    validation_data=(x_valid, y_valid)
)
