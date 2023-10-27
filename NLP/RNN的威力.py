import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten ,Dropout
from keras.layers import Embedding
from keras.layers import Conv1D,GlobalMaxPool1D
from keras.layers import SpatialDropout1D
#嵌入层的dropout！
from keras.layers import LSTM   #LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score,roc_curve
import pandas as pd
import matplotlib.pyplot as plt

#CNN模型的超参数如下：
#输出目录名
output_dir='model_output/LSTM'#对应新的目录储存每个训练周期之后的模型参数！

#训练
epochs=4
#LSTM更容易出现过拟合！
batch_size=128#这个参数不再拗述

#词向量空间的嵌入
n_dim=64#词向量的维数
n_unique_words=10000

#在词库中，一个token只有出现到一定的次数才能被嵌入到词向量空间中，我们选择前5000个最为常用的单词！
#也可能不是最优的

n_words_to_skip=50#前人的处理方法，前50个频率最高的都是停顿词！
max_review_length=100#设置评论最大长度！
#RNN容易出现梯度消失和梯度爆炸


pad_type=trunc_type='pre'#字符填充的位置！最好选择在评论之前，语句越靠后对模型的影响就越大！不必要的放在之前！
#后一个变量是截断的位置

drop_embed=0.2#嵌入层中使用dropout机制！

#神经网络的架构
n_dense=256#全连接层的神经元的数量！
dropout=0.5#必须的超参数，不在拗述！

#RNN-LSTM层
n_lstm=256
drop_lstm=0.2


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
#将离散的变量转换成连续，可以使用函数处理的变量！
model.add(Embedding(n_unique_words,n_dim,input_length=max_review_length))
model.add(SpatialDropout1D(drop_embed))

#   RNN层
model.add(Bidirectional(GRU(n_lstm,dropout=drop_lstm,return_sequences=True)))
model.add(Bidirectional(GRU(n_lstm,dropout=drop_lstm)))
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

model.load_weights(output_dir+"/weights.04.hdf5")
#第三个周期的参数加载到模型中！有时候不需要反复训练了，这里直接加载已经训练过的模型参数！
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
print(ydf.head(20))

#显示验证集数据y_hat的直方图！
plt.hist(y_hat)
_=plt.axvline(x=0.5,color='orange')
plt.show()

