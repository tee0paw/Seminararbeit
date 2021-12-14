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
CLUSTER_NUMBER = 10

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
    #points[0][2] = MISSING_VALUE
    #points[3][2] = MISSING_VALUE


def gmm_distribution_missing_values_from_file(filename):
    pass

def split_data_in_completes(data):
    # select only rows with complete data and save it in completes
    completes = np.array(data)
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
    return completes

def split_data_in_incompletes(data):
    incompletes = np.array(data)
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
    return incompletes

def calculate_subspace_means_safe_dimensions(row, gmm_means):
    cluster = 0
    while cluster < CLUSTER_NUMBER:
        index_unsafe_dimensions = [index for index in range(len(row)) if MISSING_VALUE == row[index]]
        subspace_means_safe_dimensions = np.delete(gmm_means[cluster],index_unsafe_dimensions, axis=0)
        cluster += 1
    return subspace_means_safe_dimensions

def calculate_subspace_covariances_safe_dimensions(row, gmm_covariances):
    cluster = 0
    while cluster < CLUSTER_NUMBER:
        index_unsafe_dimensions = [index for index in range(len(row)) if MISSING_VALUE == row[index]]
        subspace_covariances_safe_dimensions = np.delete(np.diag(gmm_covariances[cluster]), index_unsafe_dimensions, axis=1)
        subspace_covariances_safe_dimensions = np.delete(subspace_covariances_safe_dimensions, index_unsafe_dimensions, axis=0)
        cluster += 1
    return subspace_covariances_safe_dimensions

def calculate_subspace_means_unsafe_dimensions(row, gmm_means):
    cluster = 0
    while cluster < CLUSTER_NUMBER:
        index_safe_dimensions = [index for index in range(len(row)) if MISSING_VALUE != row[index]]
        # neuer Erwartungswert
        subspace_means_unsafe_dimensions = np.delete(gmm_means[cluster], index_safe_dimensions,axis=0)
        cluster += 1
    return subspace_means_unsafe_dimensions

def calculate_subspace_covariances_unsafe_dimensions(row, gmm_covariances):
    cluster = 0
    while cluster < CLUSTER_NUMBER:
        index_safe_dimensions = [index for index in range(len(row)) if MISSING_VALUE != row[index]]
        # neue Kovarianzmatrix
        subspace_covariances_unsafe_dimensions = np.delete(np.diag(gmm_covariances[cluster]), index_safe_dimensions, axis=1)
        subspace_covariances_unsafe_dimensions = np.delete(subspace_covariances_unsafe_dimensions, index_safe_dimensions, axis=0)
        cluster += 1
    return subspace_covariances_unsafe_dimensions

def calculate_new_means(row, gmm_means):
    cluster = 0
    new_means = np.zeros((1,np.count_nonzero(row == MISSING_VALUE)))
    while cluster < CLUSTER_NUMBER:
        index_safe_dimensions = [index for index in range(len(row)) if MISSING_VALUE != row[index]]
        # neuer Erwartungswert
        subspace_means_unsafe_dimensions = np.delete(gmm_means[cluster], index_safe_dimensions,axis=0)
        new_means = np.append(new_means, [subspace_means_unsafe_dimensions], axis=0)
        cluster += 1
    new_means = np.delete(new_means,0, axis=0)
    return new_means

def calculate_new_covariances(row, gmm_covariances):
    cluster = 0
    new_covariances = np.zeros((1,np.count_nonzero(row == MISSING_VALUE),np.count_nonzero(row == MISSING_VALUE)))
    while cluster < CLUSTER_NUMBER:
        index_safe_dimensions = [index for index in range(len(row)) if MISSING_VALUE != row[index]]
        # neue Kovarianzmatrix
        subspace_covariances_unsafe_dimensions = np.delete(np.diag(gmm_covariances[cluster]), index_safe_dimensions, axis=1)
        subspace_covariances_unsafe_dimensions = np.delete(subspace_covariances_unsafe_dimensions, index_safe_dimensions, axis=0)
        new_covariances = np.append(new_covariances, [subspace_covariances_unsafe_dimensions], axis=0)
        cluster += 1
    new_covariances = np.delete(new_covariances,0, axis=0)
    return new_covariances

def get_safe_dimensions(row):
    cluster = 0
    while cluster < CLUSTER_NUMBER:
        safe_dimensions = [value for value in row if MISSING_VALUE != value]
        safe_dimensions = np.array(safe_dimensions)
        cluster += 1
    return safe_dimensions

def calculate_c(row, gmm_means, gmm_covariances):
    cluster = 0
    c_list = np.zeros([CLUSTER_NUMBER])
    while cluster < CLUSTER_NUMBER:
        safe_dimensions = [value for value in row if MISSING_VALUE != value]
        safe_dimensions = np.array(safe_dimensions)

        index_unsafe_dimensions = [index for index in range(len(row)) if MISSING_VALUE == row[index]]
        subspace_means_safe_dimensions = np.delete(gmm_means[cluster],index_unsafe_dimensions, axis=0)

        index_unsafe_dimensions = [index for index in range(len(row)) if MISSING_VALUE == row[index]]
        subspace_covariances_safe_dimensions = np.delete(np.diag(gmm_covariances[cluster]), index_unsafe_dimensions, axis=1)
        subspace_covariances_safe_dimensions = np.delete(subspace_covariances_safe_dimensions, index_unsafe_dimensions, axis=0)
    
        c_i = multivariate_normal.pdf(safe_dimensions, mean=subspace_means_safe_dimensions, cov=subspace_covariances_safe_dimensions)
        c_list = np.append(c_list,c_i)
        cluster += 1
    c_list = c_list[c_list != 0]
    return c_list

def calculate_new_weights(c_list, gmm_weights):
    multiplicated_list = [c_list*gmm_weights for c_list,gmm_weights in zip(c_list,gmm_weights)]
    sum_multiplicated_list = sum(multiplicated_list)

    # normalize known dimensions
    cluster = 0
    new_weights = np.zeros([CLUSTER_NUMBER])

    while cluster < CLUSTER_NUMBER:
        r_i = ((c_list[cluster]*gmm_weights[cluster]) / sum_multiplicated_list)
        new_weights = np.append(new_weights,r_i)
        cluster += 1
    # neue Gewichte
    new_weights = new_weights[new_weights != 0]
    return new_weights

### TEST
def y_in_new_gmm(unsafe_dimensions,gmm_means,gmm_covariances, row, new_weights):
    cluster = 0
    new_normal_distribution_list = np.zeros([CLUSTER_NUMBER])
    density_values = np.zeros([CLUSTER_NUMBER])
    while cluster < CLUSTER_NUMBER:
        index_safe_dimensions = [index for index in range(len(row)) if MISSING_VALUE != row[index]]
        subspace_means_unsafe_dimensions = np.delete(gmm_means[cluster], index_safe_dimensions,axis=0)

        index_safe_dimensions = [index for index in range(len(row)) if MISSING_VALUE != row[index]]
        subspace_covariances_unsafe_dimensions = np.delete(np.diag(gmm_covariances[cluster]), index_safe_dimensions, axis=1)
        subspace_covariances_unsafe_dimensions = np.delete(subspace_covariances_unsafe_dimensions, index_safe_dimensions, axis=0)

        new_normal_distribution = multivariate_normal.pdf(unsafe_dimensions, mean=subspace_means_unsafe_dimensions, cov=subspace_covariances_unsafe_dimensions)
        new_normal_distribution_list = np.append(new_normal_distribution_list, new_normal_distribution)

        cluster += 1
    
    new_normal_distribution_list = new_normal_distribution_list[new_normal_distribution_list != 0]
    r_and_new_value = [new_weights*new_normal_distribution_list for new_weights,new_normal_distribution_list in zip(new_weights,new_normal_distribution_list)]
    sum_values = sum(r_and_new_value)
    
    i = 0
    while i < CLUSTER_NUMBER:
        normalized_new_normal_distribution = ((r_and_new_value[i]) / sum_values)
        density_values = np.append(density_values,normalized_new_normal_distribution)
        i += 1
    density_values = density_values[density_values != 0]
    print(density_values)
    return density_values

completes = split_data_in_completes(points)
gmm = GaussianMixture(covariance_type="diag", n_components=CLUSTER_NUMBER).fit(
        completes)

def gmm_distribution_missing_values(points):
    means_covariances_weights_list = []
    completes = split_data_in_completes(points)

    gmm = GaussianMixture(covariance_type="diag", n_components=CLUSTER_NUMBER).fit(
        completes)
    gmm_means = np.array(gmm.means_)
    gmm_covariances = np.array(gmm.covariances_)
    gmm_weights = np.array(gmm.weights_)

    ### TEST
    #probabilites = gmm.predict_proba([[-64,-56,-61,-66,-71,-82,-81]])
    #print("Verteilungen mit vollständigem Wert durch GMM predicted (-64,-56,-61,-66,-71,-82,-81): ",probabilites)

    incompletes = split_data_in_incompletes(points)

    for row in incompletes:
        subspace_means_safe_dimensions = calculate_subspace_means_safe_dimensions(row, gmm_means)
        subspace_covariances_safe_dimensions = calculate_subspace_covariances_safe_dimensions(row, gmm_covariances)
        subspace_means_unsafe_dimensions = calculate_subspace_means_unsafe_dimensions(row, gmm_means)
        subspace_covariances_unsafe_dimensions = calculate_subspace_covariances_unsafe_dimensions(row, gmm_covariances)
        new_means = calculate_new_means(row, gmm_means)
        new_covariances = calculate_new_covariances(row, gmm_covariances)
        safe_dimensions = get_safe_dimensions(row)
        c_list = calculate_c(row, gmm_means, gmm_covariances)
        new_weights = calculate_new_weights(c_list,gmm_weights)
        means_covariances_weights_one_row = new_means, new_covariances, new_weights
        means_covariances_weights_list.append(means_covariances_weights_one_row)

    ### TEST
    #new_normal_distribution_list = y_in_new_gmm([-64], gmm_means, gmm_covariances, row, new_weights)
    #print(new_normal_distribution_list)
    return means_covariances_weights_list

list_with_all_new_values = gmm_distribution_missing_values(points)
