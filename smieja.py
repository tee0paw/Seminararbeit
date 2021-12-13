import random
import sys
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN
import pylab
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal


MISSING_VALUE = 1000000


# preprocess data
with open(sys.argv[1]) as f:
    file = f.readlines()
data = []
for row in file:
    data.append(([int(item) for item in row.split("\t")[:-1]],
                 int(row.split("\t")[-1].split("\n")[0])-1))
random.shuffle(data)
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
    points[0][3] = MISSING_VALUE


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
#print("erste Dimension ist unsicher. Restliche Dimensionen sind sicher.")
#print("-56,-61,-66,-71,-82,-81 sind die sicheren Werte")
#print("-64 ist nicht bekannt")
# feet the GMM model with complete data
cluster_number=5
gmm = GaussianMixture(covariance_type="diag", n_components=cluster_number).fit(
    completes)
probabilites = gmm.predict_proba([[-64,-56,-61,-66,-71,-82,-81]])
#print("Verteilungen mit vollständigem Wert durch GMM predicted (-64,-56,-61,-66,-71,-82,-81): ",probabilites)

# save means and covariances and weights
gmm_means = np.array(gmm.means_)
gmm_covariances = np.array(gmm.covariances_)
gmm_weights = np.array(gmm.weights_)

# select only rows with incomplete data and save it in incompletes
incompletes = np.array(points)
row = 0
while row < len(incompletes):
    column = 0
    delete = True
    # Search for missing value
    while column < len(incompletes[row]):
        if incompletes[row][column] == MISSING_VALUE:
            delete = False
        column += 1
    # If there was a missing value found, delete the row
    if delete:
        incompletes = np.delete(
            incompletes, row, 0)
        row -= 1
    row += 1

for row in incompletes:
    cluster = 0
    c_list = np.zeros([cluster_number])
    new_normal_distribution_list = np.zeros([cluster_number])
    new_means = np.zeros((1,np.count_nonzero(row == MISSING_VALUE)))
    new_covariances = np.zeros((1,np.count_nonzero(row == MISSING_VALUE),np.count_nonzero(row == MISSING_VALUE)))
    while cluster < cluster_number:
        # split dimensions in safe dimensions and save the index
        safe_dimensions = [value for value in row if MISSING_VALUE != value]
        safe_dimensions = np.array(safe_dimensions)
        index_safe_dimensions = [index for index in range(len(row)) if MISSING_VALUE != row[index]]
        index_unsafe_dimensions = [index for index in range(len(row)) if MISSING_VALUE == row[index]]
   
        # split means and covariances in safe and unsafe vectors
        subspace_means_safe_dimensions = np.delete(gmm_means[cluster],index_unsafe_dimensions, axis=0)
        subspace_covariances_safe_dimensions = np.delete(np.diag(gmm_covariances[cluster]), index_unsafe_dimensions, axis=1)
        subspace_covariances_safe_dimensions = np.delete(subspace_covariances_safe_dimensions, index_unsafe_dimensions, axis=0)

        # neuer Erwartungswert
        subspace_means_unsafe_dimensions = np.delete(gmm_means[cluster], index_safe_dimensions,axis=0)
        new_means = np.append(new_means, [subspace_means_unsafe_dimensions], axis=0)
        # neue Kovarianzmatrix
        subspace_covariances_unsafe_dimensions = np.delete(np.diag(gmm_covariances[cluster]), index_safe_dimensions, axis=1)
        subspace_covariances_unsafe_dimensions = np.delete(subspace_covariances_unsafe_dimensions, index_safe_dimensions, axis=0)
        new_covariances = np.append(new_covariances, [subspace_covariances_unsafe_dimensions], axis=0)

        # calculate c_i
        c_i = multivariate_normal.pdf(safe_dimensions, mean=subspace_means_safe_dimensions, cov=subspace_covariances_safe_dimensions)
        c_list = np.append(c_list,c_i)

        # Test 
        new_normal_distribution = multivariate_normal.pdf([-64,-66], mean=subspace_means_unsafe_dimensions, cov=subspace_covariances_unsafe_dimensions)
        new_normal_distribution_list = np.append(new_normal_distribution_list, new_normal_distribution)
        
        cluster += 1

    # c multiplicated with weights
    c_list = c_list[c_list != 0]
    new_normal_distribution_list = new_normal_distribution_list[new_normal_distribution_list != 0]
    new_means = np.delete(new_means,0, axis=0)
    new_covariances = np.delete(new_covariances,0, axis=0)

    multiplicated_list = [c_list*gmm_weights for c_list,gmm_weights in zip(c_list,gmm_weights)]
    sum_multiplicated_list = sum(multiplicated_list)

    # normalize known dimensions
    i = 0
    r_list = np.zeros([cluster_number])
    n_list = np.zeros([cluster_number])

    while i < cluster_number:
        r_i = ((c_list[i]*gmm_weights[i]) / sum_multiplicated_list)
        r_list = np.append(r_list,r_i)
        i += 1
    # neue Gewichte
    r_list = r_list[r_list != 0]

    # Testwert y mit bekannten Dimensionen verrechnet
    r_and_new_value = [r_list*new_normal_distribution_list for r_list,new_normal_distribution_list in zip(r_list,new_normal_distribution_list)]
    sum_values = sum(r_and_new_value)

    # normalize unknown dimensions
    i = 0
    while i < cluster_number:
        normalized_new_normal_distribution = ((r_and_new_value[i]) / sum_values)
        n_list = np.append(n_list,normalized_new_normal_distribution)
        i += 1
    n_list = n_list[n_list != 0]
    #print("neue Verteilungen (wenn fehlende Dimension eingesetzt wird in neues Modell): ", n_list)
    #print(sum(n_list))
    #print("c_i für die fünf Cluster:", c_list)
    #print("Anzahl an fehlenden Dimensionen: ", len(index_unsafe_dimensions))
    #print("r_i für die fünf Cluster: ", r_list)
    #print("Summe der normalisierten Gewichten: ", sum(r_list))
    #print("Neue Erwartungswerte für ", cluster_number, " Cluster: " ,new_means)
    #print("Neue Kovarianzmatrizen für ", cluster_number, " Cluster: ",new_covariances)

# check accurancy of the predicted cluster with the heighest probability
    max_weight = 0
    i = 0
    while i < len(probabilites[0]):
        if (probabilites[0][i] > max_weight):
            max_weight = probabilites[0][i]
            best_cluster = np.where(probabilites[0] == max_weight)
        i += 1
    print("Wahrscheinlichkeit: ", max_weight, "Cluster: ", best_cluster)

    max_weight = 0
    i = 0
    while i < len(n_list):
        if (n_list[i] > max_weight):
            max_weight = n_list[i]
            best_cluster = np.where(n_list == max_weight)
        i += 1
    print("Wahrscheinlichkeit: ", max_weight, "Cluster: ", best_cluster)

