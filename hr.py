#KNN or "simple majority vote classification"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.model_selection import train_test_split

#importing dataset
fruits = pd.read_table(r"C:\Users\Bilguun.NOMLOG\Desktop\Coursera-ML\.vscode\KNN classifier\fruit_data_with_colors.txt")

X = fruits[["color_score", "width", "height"]]
y = fruits["fruit_label"]

#splitting data into training and test sets
#here random_state initializes the seed state
#same seed value results in same randomized value
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# #scatterplot of the data
# cmap = cm.get_cmap("gnuplot")
# scatter = pd.plotting.scatter_matrix(X_train, c = y_train, marker="o", s = 40, hist_kwds = {"bins":15}, figsize=(12, 12), cmap = cmap)

# #3d plot
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = "3d")
# ax.scatter(X_train["width"], X_train["height"], X_train["color_score"], c = y_train, marker = "o", s = 100)
# ax.set_xlabel("width")
# ax.set_ylabel("height")
# ax.set_zlabel("color_score")
# plt.show()

lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

#KNN 
from sklearn.neighbors import KNeighborsClassifier

#creating classifier object
knn = KNeighborsClassifier(n_neighbors=5)

#training the classifier using training data
knn.fit(X_train, y_train)

#estimating accuracy using test data
knn.score(X_test, y_test)

#sample prediction
fruit_prediction = knn.predict([0.79, 4.3, 5.5])
print(lookup_fruit_name[fruit_prediction[0]])

from adspy_shared_utilities import plot_fruit_knn
plot_fruit_knn(X_train, y_train, 5, "uniform")