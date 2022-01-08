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
    int(len(points) / 4), int(len(points) / 4))

for i in range(number_of_deleted_values):
    position_of_array_deleted_in_array = random.randint(0, len(points) - 1)
    position_of_value_deleted_in_array = random.randint(
        0, len(points[position_of_array_deleted_in_array]) - 1)
    # points[position_of_array_deleted_in_array][position_of_value_deleted_in_array] = MISSING_VALUE
    points[0][0] = MISSING_VALUE

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

# feet the GMM model with complete data
gmm = GaussianMixture(covariance_type="full", n_components=5).fit(
    completes)
# First row in the data 
predicted_correct_value = gmm.predict([[-64,-56,-61,-66,-71,-82,-81]])
predicted_correct_value_density = gmm.predict_proba([[-64,-56,-61,-66,-71,-82,-81]])
print("Tatsächliches Cluster: ",predicted_correct_value)
print("Tatsächliche Wahrscheinlichkeit für Einordnung in Cluster: ", predicted_correct_value_density)
# Convert means of the GMM to int
gmm_means = np.round_(gmm.means_,0)
gmm_means = gmm_means.astype(int)

# select only rows with incomplete data and save it in incompletes
incompletes = np.array(points)
row = 0
while row < len(incompletes):
    column = 0
    delete = True
    # Search for missing value
    while column < len(incompletes[row]):
        if incompletes[row][column] == MISSING_VALUE:
            # Replace missing value by the mean of each cluster for the missing dimension and saves the heighest weight
            i = 0
            max_weight = 0.0
            best_cluster = 0            
            while i < len(gmm.means_):
                incompletes[row][column] = gmm_means[i][column]
                gmm_weights = gmm.predict_proba([incompletes[i]])[0]
                for cluster, weight in enumerate(gmm_weights, start=0):
                    print("Wert: ", incompletes[row])
                    print("cluster: ", cluster, " Wahrscheinlichkeit: ", weight)
                    if weight > max_weight:
                        max_weight = weight
                        best_cluster = cluster
                i += 1
            print("Wahrscheinlichkeit: ", max_weight, "Cluster: ", best_cluster)
            delete = False
        column += 1
    # If there was a missing value found, delete the row
    if delete:
        incompletes = np.delete(
            incompletes, row, 0)
        row -= 1
    row += 1

sys.exit()

gmm = GaussianMixture(covariance_type="full", n_components=5).fit(
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
