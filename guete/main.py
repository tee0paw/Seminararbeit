import matplotlib.pyplot as plt
import pylab
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

with open("wifi_localization.txt") as f:
    file = f.readlines()
data = []
for row in file:
    data.append(([int(item) for item in row.split("\t")[:-1]], int(row.split("\t")[-1].split("\n")[0])-1))
import random
random.shuffle(data)
points = np.array([item[0] for item in data])
pointsDeletedList = []
for pointRow in points:
    pointRow = np.delete(pointRow, 0)
    pointsDeletedList.append(pointRow)
pointsDeleted = np.array(pointsDeletedList)
print(pointsDeleted)

gmm = GaussianMixture(covariance_type= "full", n_components=20).fit(points[:500])
pca = PCA(n_components=2)
pca.fit(gmm.means_)
points_transformed = pca.transform(points).tolist()[500:]
means_transformed = pca.transform(gmm.means_).tolist()

fig = plt.figure()
ax = fig.add_subplot()

ax.scatter([item[0] for item in points_transformed], [item[1] for item in points_transformed])
ax.scatter([item[0] for item in means_transformed], [item[1] for item in means_transformed])

pylab.show()



