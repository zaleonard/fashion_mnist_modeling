#%%
import pandas as pd
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pickle



#preprocessing if images in folders
#not necessary for fashion_MNIST data

# input_dir = 'path'
# categories = [
#     '0', '1', '2', '3', '4', '5',
#     '6', '7', '8', '9'
# ]

# data=[]
# labels=[]
# for category_idx, category in enumerate(categories):
#     for file in os.listdir(os.path.join(input_dir, category)):
#         img_path = os.path.join(input_dir, category, file)
#         img = imread(img_path)
#         img = resize(img, (28,28))
#         data.append(img.flatten()) # make the image into 1 array for data
#         labels.append(category_idx)

# data = np.asarray(data)
# labels = np.asarray(labels)

data_train = pd.read_csv(r'C:\Users\zaleo\Desktop\AI\cv_mnist\fashion-mnist_train.csv')
data_test = pd.read_csv(r'C:\Users\zaleo\Desktop\AI\cv_mnist\fashion-mnist_test.csv')

print(data_train.head())

#%%

#sk learn modeling
data_train_labels = data_train['label']
data_train = data_train.drop('label', axis=1)

data_test_labels = data_test['label']
data_test = data_test.drop('label', axis=1)


#train / test 
minmax_scaler = preprocessing.MinMaxScaler()
min_max_data = minmax_scaler.fit_transform(data_train)
min_max_data = pd.DataFrame(
    min_max_data, columns=data_train.columns
)

# standard_scaler = preprocessing.StandardScaler()
# standard_data = standard_scaler.fit_transform(data_train)
# standard_data = pd.DataFrame(standard_data, columns=data_train.columns)

# robust_scaler = preprocessing.RobustScaler()
# robust_data = robust_scaler.fit_transform(data_train)
# robust_data = pd.DataFrame(
#     robust_data, columns=data_train.columns
# )

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(15, 5))

# for col in data_train:
#     ax1.set_title("Before Scaling")
#     sns.kdeplot(data_train[col], ax=ax1)
# for col in data_train:
#     ax2.set_title("After Min-Max Scaling")
#     sns.kdeplot(min_max_data[col], ax=ax2)
# for col in data_train:
#     ax3.set_title("After Standard Scaling")
#     sns.kdeplot(standard_data[col], ax=ax3)
# for col in data_train:
#     ax4.set_title("After Robust Scaling")
#     sns.kdeplot(robust_data[col], ax=ax4)
# plt.show()

#min-max scaling looks most appropriate, PCA analysis now
pca = PCA()
pca.fit(min_max_data)

#determine number of components needed to explain 95% of variance
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Create a scree plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.legend(loc='best')
plt.grid(True)

pca=PCA(n_components=188)
X_pca = pca.fit_transform(min_max_data)
data_train_pca = pd.DataFrame(data=X_pca)



#%%
x_train, x_test, y_train, y_test = train_test_split(
    data_train_pca, data_train_labels,
    test_size = 0.2,
    stratify = data_train_labels,
    shuffle=True
    )
#been having issues with long training sessions
#implementing earlystopping to speed up hyperparemter tuning

class EarlyStoppingSVC(SVC):
    def fit(self, X, y):
        super().fit(X, y)
        return self
     
classifier = EarlyStoppingSVC()
#classifier training
parameters = [
    {'gamma' : [0.01, 0.001],
     'C': [1, 10]}
    ]

random_search = RandomizedSearchCV(classifier, parameters, n_iter=10, random_state=42)
random_search.fit(x_train, y_train)

#performance metrics
best_estimator = random_search.best_estimator_
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print(best_estimator)
print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))
#
