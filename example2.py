import random
import sys
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN
import pylab
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


MISSING_VALUE = 1000000


# preprocess data
with open(sys.argv[1]) as f:
    file = f.readlines()
data = []
for row in file:
    data.append(([int(item) for item in row.split("\t")[:-1]],
                 int(row.split("\t")[-1].split("\n")[0])-1))
#random.shuffle(data)
points = np.array([item[0] for item in data])

# delete random values in points between 0 and 25%
number_of_deleted_values = random.randint(
    0, int(len(points) / 4))

for i in range(number_of_deleted_values):
    position_of_array_deleted_in_array = random.randint(0, len(points) - 1)
    position_of_value_deleted_in_array = random.randint(
        0, len(points[position_of_array_deleted_in_array]) - 1)
    points[position_of_array_deleted_in_array][position_of_value_deleted_in_array] = MISSING_VALUE

# *** use this if you want to check the prediction of single rows (second value of second row is missing)
#position_of_array_deleted_in_array = 1
#position_of_value_deleted_in_array = 1
#points[position_of_array_deleted_in_array][position_of_value_deleted_in_array] = MISSING_VALUE

# select only rows with complete data and save it in completes
completes = np.array(points)
row = 0
while row < len(completes):
    column = 0
    while column < len(completes[row]):
        if completes[row][column] == MISSING_VALUE:
            completes = np.delete(
                completes, row, 0)
            row -= 1
        column += 1
    row += 1

# calculate mean of every column and convert it to int
completes_mean = completes.mean(axis=0)
completes_mean = np.round_(completes_mean,0)
completes_mean = completes_mean.astype(int)

# feet the GMM model with complete data
gmm = GaussianMixture(covariance_type="full", n_components=10).fit(
    completes)

# select only rows with incomplete data and save it in incompletes
incompletes = np.array(points)
row = 0
while row < len(incompletes):
    column = 0
    delete = True
    # Search for missing value
    while column < len(incompletes[row]):
        if incompletes[row][column] == MISSING_VALUE:
            # Replace missing values by the mean of the dimension
            incompletes[row][column] = completes_mean[column]
            delete = False
        column += 1
    # If there was a missing value found, delete the row
    if delete:
        incompletes = np.delete(
            incompletes, row, 0)
        row -= 1
    row += 1
# *** this section checks the prediction of the first row which was deleted in row 34 in the script
#predicted_correct_value = gmm.predict([[-68,-57,-61,-65,-71,-85,-85]])
#predicted_missing_value = gmm.predict(incompletes)
#predicted_prob = gmm.predict_proba(incompletes)
#print("predicted missing value", predicted_missing_value)
#print("predicted correct value", predicted_correct_value)
sys.exit()


gmm = GaussianMixture(covariance_type="full", n_components=10).fit(
    completes)
pca = PCA(n_components=2)
pca.fit(gmm.means_)
points_transformed = pca.transform(completes).tolist()
means_transformed = pca.transform(gmm.means_).tolist()

fig = plt.figure()
ax = fig.add_subplot()

ax.scatter([item[0] for item in points_transformed], [item[1]
                                                      for item in points_transformed])
ax.scatter([item[0] for item in means_transformed], [item[1]
                                                     for item in means_transformed])

pylab.show()
