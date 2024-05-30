import pandas as pd
# made by bowen xu

# 1. 数据预处理
# 加载数据
data_path = 'heart_statlog_cleveland_hungary_final.csv'
data = pd.read_csv(data_path)

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 初始化标签编码器
le = LabelEncoder()

# 对分类变量进行标签编码
categorical_features = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']
for col in categorical_features:
    data[col] = le.fit_transform(data[col])


# 划分数据集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 提取标签
y_train = train_data['target'].values
y_test = test_data['target'].values

# 移除不必要的列
train_data = train_data.drop(columns=['target'])
test_data = test_data.drop(columns=['target'])

# 标准化特征（PCA前的常见做法）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data)
X_test_scaled = scaler.transform(test_data)

# 2. PCA
# 应用PCA
pca = PCA(n_components=2) # 只保留前两个主成分
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 绘制主成分阶梯图
import matplotlib.pyplot as plt

variance_explained = pca.explained_variance_ratio_
cumulative_variance = variance_explained.cumsum()

plt.figure(figsize=(8, 4))
plt.bar(range(1, len(variance_explained)+1), variance_explained, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1, len(cumulative_variance)+1), cumulative_variance, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.title('Scree Plot')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 输出主成分的方差解释比例
import numpy as np

def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    
    plt.figure(figsize=(8, 8))
    plt.scatter(xs * scalex, ys * scaley,  c=['blue' if label == 0 else 'skyblue' for label in y_train])  
    for i in range(len(coeff)):
        x, y = coeff[i, 0], coeff[i, 1]
        plt.arrow(0, 0, x, y, color='black', alpha=1)
        if labels is None:
            plt.text(x * 1.15, y * 1.15, "Var"+str(i+1), color='black', ha='center', va='center', fontsize=15)
        else:
            plt.text(x * 1.15, y * 1.15, labels[i], color='black', ha='center', va='center', fontsize=15)
    
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.title('Biplot')
    plt.show()

features = train_data.columns
biplot(X_train_pca, pca.components_, labels=features)

# 3. 逻辑回归
from sklearn.linear_model import LogisticRegression
model_1 = LogisticRegression()
model_1.fit(X_train_pca, y_train)
print(model_1.score(X_test_pca, y_test))

# 4. 绘制决策边界
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, alpha=0.3, levels=[0, 0.5, 1], cmap='RdBu')
    plt.contour(xx, yy, preds, levels=[0.5], colors='purple')

    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='#DA70D6', marker='o', edgecolors='k', label='No Heart Disease')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='#0000CD', marker='x', label='Heart Disease')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(False)
    plt.show()

# 绘制训练集和测试集的决策边界
plot_decision_boundary(X_train_pca, y_train, model_1)
plot_decision_boundary(X_test_pca, y_test, model_1)




