import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = 'D:\code\RUClearn\shukegaiproject\skg new_version\新数科概（修改配色版）\heart_statlog_cleveland_hungary_final.csv'
data = pd.read_csv(file_path)

sns.set(style='whitegrid')

#绘制年龄分布图
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=30, kde=True)
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

for item in data['resting bp s']:
    if item < 100:
        print(item)

#绘制静息血压与胆固醇水平比较散点图
plt.figure(figsize = (10, 6))
sns.scatterplot(x= 'resting bp s', y = 'cholesterol', hue = 'target', data = data)
plt.title('Resting blood pressure and Cholesterol levels')
plt.xlabel('Resting blood pressure')
plt.ylabel('Cholesterol')
plt.legend(title='Heart disease')
plt.show()

#绘制胸痛程度分部图
plt.figure(figsize=(10, 6))
sns.countplot(x = 'chest pain type', data = data)
plt.title('Distribution of chest pain types')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.show()

#运动心绞痛与最大心率箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x='exercise angina', y='max heart rate', data=data, palette='muted')
plt.title('Maximum heart rate by exercise induced angina')
plt.xlabel('Exercise induced angina')
plt.ylabel('Maximum heart rate')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()
 
#相关性热力图
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation heatmap')
plt.xticks(rotation = -30)
plt.show()


#最大心率与年龄
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='max heart rate', hue='target', data=data)
plt.title('Maximum heart rate vs age')
plt.xlabel('Age')
plt.ylabel('Maximum heart rate')
plt.legend(title='Heart disease')
plt.show()