import pandas as pd

# 创建一个简单的 DataFrame
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 70000, 80000]
})

# 创建布尔系列，筛选 age 大于 30 的行
boolean_series = df['age'] > 30
print(boolean_series)
print(df)
