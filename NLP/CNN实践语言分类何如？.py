import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten ,Dropout
from keras.layers import Embedding
from keras.layers import Conv1D,GlobalMaxPool1D#一维的——用于时序预测的卷积核就足够了！
from keras.layers import SpatialDropout1D
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score,roc_curve
import pandas as pd
import matplotlib.pyplot as plt

#CNN模型的超参数如下：
#输出目录名
output_dir='model_output/conv'#对应新的目录储存每个训练周期之后的模型参数！

#训练
epochs=4#NLP比一般的机器视觉模型更早出现过拟合！
batch_size=128#这个参数不再拗述

#词向量空间的嵌入
n_dim=64#词向量的维数
n_unique_words=5000
#在词库中，一个token只有出现到一定的次数才能被嵌入到词向量空间中，我们选择前5000个最为常用的单词！
#也可能不是最优的
n_words_to_skip=50#前人的处理方法，前50个频率最高的都是停顿词！
max_review_length=400#设置评论最大长度！
pad_type=trunc_type='pre'#字符填充的位置！最好选择在评论之前，语句越靠后对模型的影响就越大！不必要的放在之前！
#后一个变量是截断的位置

drop_embed=0.2#嵌入层中使用dropout机制！

#神经网络的架构
n_dense=256#全连接层的神经元的数量！
dropout=0.5#必须的超参数，不在拗述！

#卷积层
n_conv=256#卷积核的数量
k_conv=3#卷积核的大小


#加载影评的数据
(x_train,y_train),(x_valid,y_valid)=imdb.load_data(num_words=n_unique_words,skip_top=n_words_to_skip)
#第一个参数限制词向量词汇表的大小
#第二个参数删除常见的停顿词！


#构造单词索引
word_index=keras.datasets.imdb.get_word_index()
word_index={k:(v+3) for k,v in word_index.items()}
word_index["PAD"]=0
word_index["START"]=1
word_index["UNK"]=2
index_word={v:k for k,v in word_index.items()}

#' '.join(index_word[id] for id in x_train[0])
(all_x_train,_),(all_x_valid,_)=imdb.load_data()
' '.join(index_word[id] for id in all_x_train[0])

#标准化评论长度！
x_train=pad_sequences(x_train,maxlen=max_review_length,padding=pad_type,truncating=trunc_type,value=0)
x_valid=pad_sequences(x_valid,maxlen=max_review_length,padding=pad_type,truncating=trunc_type,value=0)

model=Sequential()
#嵌入层
model.add(Embedding(n_unique_words,n_dim,input_length=max_review_length))
model.add(SpatialDropout1D(drop_embed))

#卷积层
model.add(Conv1D(n_conv,k_conv,activation='relu'))
model.add(Conv1D(n_conv,k_conv,activation='relu'))
model.add(GlobalMaxPool1D())#全局最大池化层！

#全连接层
model.add(Dense(n_dense,activation='relu'))
model.add(Dropout(dropout))

#输出层
model.add(Dense(1,activation='sigmoid'))

#编译模型
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
modelcheckpiont=ModelCheckpoint(filepath=output_dir+"/weights.{epoch:02d}.hdf5")
#每个训练周期完成后保留参数！

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#训练全连接情感分类器！
model.fit(x_train,y_train,
          batch_size=batch_size,epochs=epochs,verbose=1,
          validation_data=(x_valid,y_valid),
          callbacks=[modelcheckpiont]
          )

model.load_weights(output_dir+"/weights.03.hdf5")
#第三个周期的参数加载到模型中！
#其在验证集上的损失最低！

#正向传播得到预测值
y_hat=model.predict(x_valid)
# 比较验证模型预测褒贬的能力如何！print(y_valid[0],'\n',y_hat[0])

#计算验证数据集的ROC AUC指标
pct_auc = roc_auc_score(y_valid, y_hat) * 100.0
formatted_auc = "{:0.2f}".format(pct_auc)
print(formatted_auc)

#创建由y_valid 和y_hat组成的数据表ydf
float_y_hat=[]
for y in y_hat:
    float_y_hat.append(y[0])
ydf=pd.DataFrame(list(zip(float_y_hat,y_valid)),columns=['y_hat','y'])
print(ydf.head(10))

#显示验证集数据y_hat的直方图！
plt.hist(y_hat)
_=plt.axvline(x=0.5,color='orange')
plt.show()

#因为我们使用了卷积，矩阵的计算带来的巨大计算量使单次训练时间大大增加！但是相对于全连接网络，其带来的巨大的提升是值得
#全连接网络训练无论多少次，超参数无论怎么调试，都无法达到的！