import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(a)
print(a >= 7)
print(np.dtype(np.int8))
print(np.sum(a ** 2))
certificates_earned = pd.Series(
    [8, 2, 5, 6],
    index=['Tom', 'Kris', 'Ahmad', 'Beau']
)
print(certificates_earned[certificates_earned > 5])
print(certificates_earned.mean())
print('\n')
certificates_earned1 = pd.DataFrame({
    'Certificates': [8, 2, 5, 6],
    'Time (in months)': [16, 5, 9, 12],
    'age': [18, 19, 20, 21]
})
names = ['Tom', 'Kris', 'Ahmad', 'Beau']
certificates_earned1.index = names
major = pd.Series(
    ['Computer Science', 'Computer Engineering', 'Artificial Intelligence', 'Data Engineering', ],
    index=names,
    name='Major'
)
print(major)
certificates_earned1['Major'] = major
print(certificates_earned1)
print(certificates_earned1.shape)
data = pd.read_excel(
    'D:/competition/2024数学建模国赛/2011B/cumcm2011B附件2_全市六区交通网路和平台设置的数据表.xls',
    sheet_name='全市交通路口节点数据',
    names=['全市路口节点标号', '路口的横坐标X', '路口的纵坐标Y', '路口所属区域', '发案率(次数)'],
    index_col=0
)
print(data.head())
print(data.tail(10))
s = pd.Series(['a', 3, np.nan, 1, np.nan])
print(s.notnull().sum())

print(np.ones((2, 3, 4)))



a = np.array(([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]))
b = (np.max(a, axis=0).sum())
print(b)

