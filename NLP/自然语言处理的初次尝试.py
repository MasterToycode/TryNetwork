import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')

import string
import matplotlib.pyplot as plt
import gensim
from gensim.models.phrases import Phraser,Phrases
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE

import pandas as pd
from bokeh.io import output_notebook,output_file
from bokeh.plotting import show,figure
from bokeh.models import ColumnDataSource, LabelSet

from nltk.corpus import gutenberg
import numpy as np
#利用内置的sents()方法实现将语料库转换成列表，并将单个句子分解为单词级别
gberg_sents=gutenberg.sents()

#将二维词组处理为一个单独的token
#用于训练一个“检测器”，以确定每个二维数组相对于每个单词“单独”出现再语料库中的频率
#phrases=Phrases(gberg_sents)
#用于获取由前者检测到的二维词组
#bigram=Phraser(phrases)

#预处理整个语料库
lower_sents=[]
for s in gberg_sents:
    lower_sents.append([w.lower() for w in s if w.lower() not in list(string.punctuation)])
#初始化一个名为lower_sents的空列表，使用for循环将预处理语句放进列表，转换成小写的时候只删去了标点

lower_bigram=Phraser(Phrases(lower_sents,min_count=32,threshold=64))

clean_sents=[]
for s in lower_sents:
    clean_sents.append(lower_bigram[s])
#创建一个只包含二元词组的“干净”语料库！

model=Word2Vec(sentences=clean_sents,vector_size=64,sg=1,window=10,min_count=10,workers=4,epochs=5
    )
#word2vec作为当今最流行的构造词向量的模型之一，SG，CBOW的另种基本框架！SG根据目标单词预测上下文单词！，CBOW相反。
#这个模型关注语义，对于语法不太关注！
#vector_size维词向量空间的维数 sg=1，表示我们选择SG这个基本框架，sg=0,默认选择CBOW这个基本框架。
#window表示滑动窗口的大小，我们选择适合小型语料库的SG剧本框架，
#iter设置我们=滑过所有单词的次数！  类似我们在训练CNN时训练多个周期！
#min_count我们设置在这个语料库中的单词出现最小次数，小于这个频率我们不再训练这个单词！
#workers为并行工作数，例如CPU为8个核，这个超参数不能超过8个！
model.save('clean_gutenberg_model.w2v')
model=Word2Vec.load('clean_gutenberg_model.w2v')




#每次运行这个模型，单词的位置都是随机分配的！多次运行查找同一个单词在词向量中的位置会输出不同的结果！
print(model.wv['dog'])#输出‘dog’这个单词在词向量空间中的位置！
print(model.wv.most_similar('father',topn=3))
print(model.wv.doesnt_match("mother father sister brother dog".split()))
#找出这几个单词中某个单词和其他单词相关度最低的
# 计算'father'和'dog'的相似度
similarity_score = model.wv.similarity('father', 'dog')
print("相似度分数：", similarity_score)
print('\n')



# 查找与'father'和'woman'相关，但不相关于'man'的单词
similar_words = model.wv.most_similar(positive=['father', 'woman'], negative=['man'])
for word, score in similar_words:
    print(word, score)
print('\n')




# 查找与'father'和'woman'相关，但不相关于'man'的单词
similar_words = model.wv.most_similar(positive=['husband', 'woman'], negative=['man'])
for word, score in similar_words:
    print(word, score)




# 创建一个t-SNE对象
tsne = TSNE(n_components=2, n_iter=250)
vocab = list(model.wv.key_to_index.keys())
# 获取词汇向量
vectors = [model.wv[word] for word in vocab]
# 使用t-SNE降维
vectors_array = np.array(vectors)
X_2d = tsne.fit_transform(vectors_array)
# 创建一个DataFrame来存储坐标和词汇
coords_df = pd.DataFrame(X_2d, columns=['x', 'y'])
coords_df['tokn']=vocab
# 显示前几行数据
print(coords_df.head())

'''
# 使用matplotlib创建散点图，不要传递figure参数
plt.figure(figsize=(12, 12))  # 指定图形的尺寸
plt.scatter(coords_df['x'], coords_df['y'], marker='.', s=10, alpha=0.2)  # 创建散点图
plt.show()  # 显示图形
'''

# 创建一个子集以减小数据量
subset_df = coords_df.sample(n=5000)
# 创建一个Bokeh图形，设置宽度和高度
p = figure(width=800, height=800)
# 创建一个ColumnDataSource，用于传递数据给Bokeh
source = ColumnDataSource(subset_df)
# 创建散点图
p.scatter(x='x', y='y', source=source, marker='.', size=10, alpha=0.2)
# 创建标签文本
labels = LabelSet(x='x', y='y', text='token', text_align='left', text_baseline='middle', text_font_size='10px')
# 将标签文本添加到图形
p.add_layout(labels)
# 在PyCharm中显示图形
show(p)








