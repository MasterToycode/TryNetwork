import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import numpy as np

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

# 创建一个NumPy数组
my_array = np.array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5], dtype=np.uint8)

plt.figure(figsize=(5, 5))
for k in range(12):
    plt.subplot(3, 4, k + 1)
    plt.imshow(x_train[k], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

# 编译模型，指定损失函数和优化器
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train.reshape(-1, 784), y_train, batch_size=128, epochs=500, verbose=1, validation_data=(x_valid.reshape(-1, 784), y_valid))



