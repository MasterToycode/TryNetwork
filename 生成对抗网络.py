#用于数据输出和输入：
import numpy as np
import os

#用于深度学习：
import keras
from keras.models import Model
from keras.layers import Input,Dense,Dropout,Conv2D
from keras.layers import BatchNormalization,Flatten
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Convolution2DTranspose,UpSampling2D
from keras.optimizers import RMSprop

#用于绘图和可视化
import pandas as pd
from matplotlib import pyplot as plt

#加载数据
input_images="D://不会编程//Shiyan Neural Network//GAN//Quick Draw data//full_numpy_bitmap_apple.npy"
data=np.load(input_images)
data=data/225#和对MNIST数字集处理相同，将像素除以225，归一化操作
data=np.reshape(data,(data.shape[0],28,28,1))#判别器网络的第一个隐藏层是二维卷积层，1*784的准换成28*28*1，因为是黑白的
img_w,img_h=data.shape[1:3]#储存高度！

plt.imshow(data[4242,:,:,0],cmap="Greys")
#plt.show()



#构建判别器网络模型
def build_discriminator(depth=64,p=0.4):

    #定义输入
    image=Input((img_w,img_h,1))

    #卷积层
    conv1=Conv2D(depth*1,5,strides=2,padding='same',activation="relu")(image)

    conv1=Dropout(p)(conv1)

    conv2=Conv2D(depth*2,5,strides=2,padding='same',activation="relu")(conv1)

    conv2=Dropout(p)(conv2)

    conv3=Conv2D(depth*4,5,strides=2,padding='same',activation="relu")(conv2)

    conv3=Dropout(p)(conv3)

    conv4=Conv2D(depth*8,5,strides=1,padding='same',activation="relu")(conv3)

    conv4=Flatten()(Dropout(p)(conv4))

    #输出层
    prediction=Dense(1,activation='sigmoid')(conv4)

    #定义模型
    model=Model(inputs=image,outputs=prediction)
    return model
#卷积核大小控制为5个，步长为2*2,最后一个为1*1

discriminator=build_discriminator()

discriminator.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(learning_rate=0.0008,weight_decay=6e-8,clipvalue=1.0),
                      metrics=['accuracy']
                      )
#RMSprop理想优化器衰减率。最后一个超参数是为了防止损失函数的梯度超过这个值，防止梯度爆炸！



#构建生成器网络
z_dimensions=32
def build_generator(laten_dim=z_dimensions,depth=64,p=0.4):

    #定义输出
    noise=Input((laten_dim,))

    #第一个全连接层
    dense1=Dense(7*7*depth)(noise)
    dense1=BatchNormalization(momentum=0.9)(dense1)
    dense1=Activation(activation='relu')(dense1)
    dense1=Reshape((7,7,depth))(dense1)
    dense1=Dropout(p)(dense1)

    #反卷积层
    conv1=UpSampling2D()(dense1)
    conv1=Convolution2DTranspose(int(depth/2),kernel_size=5,padding='same',activation=None)(conv1)
    conv1=BatchNormalization(momentum=0.9)(conv1)
    conv1=Activation(activation='relu')(conv1)

    conv2=UpSampling2D()(conv1)
    conv2=Convolution2DTranspose(int(depth/4),kernel_size=5,padding='same',activation=None)(conv2)
    conv2 = BatchNormalization(momentum=0.9)(conv2)
    conv2 = Activation(activation='relu')(conv2)

    conv3 = UpSampling2D()(conv2)
    conv3 = Convolution2DTranspose(int(depth/8), kernel_size=5, padding='same', activation=None)(conv3)
    conv3 = BatchNormalization(momentum=0.9)(conv3)
    conv3 = Activation(activation='relu')(conv3)

    #输出层
    image=Conv2D(1,kernel_size=5,padding='same',activation='sigmoid')(conv3)

    #定义模型
    model=Model(inputs=noise,outputs=image)

    return model

generator=build_generator()

# 对抗和网络模型架构
z = Input(shape=(z_dimensions,))  # 输入随机噪声组
fake_img = generator(z)  # 输入生成器
discriminator.trainable = False  # 为了训练生成器，判别器的模型参数必须固定
pred = discriminator(fake_img)  # 伪造图像输入以固定的判别器模型，输入预测值
adversarial_model = Model(inputs=z, outputs=pred)


#编译对抗网络
adversarial_model.compile(loss='binary_crossentropy',
                          optimizer=RMSprop(learning_rate=0.0004,weight_decay=3e-8,clipvalue=1.0),
                          metrics=['accuracy']
                          )
#训练生成对抗网络

def train(epochs=2000,batch=12,z_dim=z_dimensions):
    d_metrics=[]
    a_metrics=[]

    running_d_lose=0
    running_d_acc=0
    running_a_lose=0
    running_a_acc=0

    for i in range(epochs):
        #真实图像
        real_things=np.reshape(
            data[np.random.choice(data.shape[0],batch,replace=False),(batch,28,28,1)]
        )

        #生成伪造图像：
        fake_things=generator.predict(
            np.random.uniform(-1.0,1.0,size=[batch,z_dim])
        )


        #将图像输入判别器：
        x=np.concatenate((real_things,fake_things))

        #用标签y表示判别器的输出结果：
        y=np.ones([2*batch,1])
        y[batch:,:]=0

        #训练判别器：
        d_metrics.append(
            discriminator.train_on_batch(x,y)
        )
        running_d_lose+=d_metrics[-1][0]
        running_d_acc+=d_metrics[-1][1]

        #将对抗网络的噪声输入和“真实值”y:
        noise=np.random.uniform(-1.0,1.0,size=[batch,z_dim])

        y=np.ones([batch,1])

        #训练对抗网络：
        a_metrics.append(
            adversarial_model.train_on_batch(noise,y)
        )

        running_a_lose+=a_metrics[-1][0]
        running_a_acc+=a_metrics[-1][1]

        #输出伪造图像：
        if (i+1)%100==0:


            print('Epoch #{}'.format(i))
            log_mesg="%d:[D loss: %f ,acc:%f]"% \
                     (i,running_d_lose/i,running_d_acc/i)
            log_mesg = "%s:[A loss: %f ,acc:%f]" % \
                       (i, running_a_lose / i, running_a_acc / i)
            print(log_mesg)

            noise=np.random.uniform(-1.0,1.0,size=[16,z_dim])

            gen_imgs=generator.predict(noise)

            plt.figure(figsize=(5,5))

            plt.show()

            for k in range(gen_imgs.shape(0)):
                plt.subplot(4,4,k+1)
                plt.imshow(gen_imgs[k,:,:,0],
                           cmap='gray'
                           )
                plt.axis('off')

            plt.tight_layout()
            plt.show()
    return a_metrics,d_metrics

a_metrics_complete,d_metrics_complete=train()