# Author: John Torres
# Auburn Email: jnt0024@auburn.edu
# Student Id: 904042457

# Libraries and Modules

import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
import sklearn.datasets
import sklearn.preprocessing
import sklearn.decomposition
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import seaborn as sns

def create_graph():

    #load data of an 8x8 image
    ds = load_digits()

    #standardize data along an axis
    dig_data = scale(ds.data)

    #filer data into unique values
    num = len(np.unique(ds.target))
    
    #pcauce dimensionality to 2 dimensions
    pca = PCA(n_components=2).fit_transform(dig_data)

    #Set boundaries for graph
    h = 0.02
    xmin, xmax = pca[:, 0].min() - 1, pca[:,0].max() + 1
    ymin, ymax = pca[:, 1].min() - 1, pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))

    # Models for each distance
    models = [
        (
            KMedoids(metric="manhattan", n_clusters=3,
                     init="heuristic", max_iter=2), "Manhattan metric",
        ),
        (
            KMedoids(metric="euclidean", n_clusters=4,
                     init="heuristic", max_iter=2), "Euclidean metric",
        ),
        (KMedoids(metric="cosine", n_clusters=5, init="heuristic",
                  max_iter=2), "Cosine metric",), ]

    # number of rows = integer(ceiling(number of model variants/2))
    num_rows = int(np.ceil(len(models) / 2.0))
    # number of columns
    num_cols = 2

    # Clear the current figure first (if any)
    plt.clf()
    # Initialize dimensions of the plot
    plt.figure(figsize=(15, 10))

    for i, (model, description) in enumerate(models):
        # Fit each point in the mesh to the model
        model.fit(pca)
        # Ppcaict the labels for points in the mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result  into a color plot
        Z = Z.reshape(xx.shape)
        # Subplot for the ith model variant
        plt.subplot(num_cols, num_rows, i + 1)
        # Display the subplot
        plt.imshow(
            Z,  # data to be plotted
            interpolation="nearest",
            # bounding box coordinates (left,right,bottom,top)
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,  # colormap
            aspect="auto",  # aspect ratio of the axes
            origin="lower",  # set origin as lower left corner of the axes
        )
        plt.plot(
            pca[:, 0], pca[:, 1], "k.", markersize=2, alpha=0.3
        )
        # Plot the centroids as white cross marks
        centroids = model.cluster_centers_
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,  # marker’s size (points^2)
            linewidths=3,  # width of boundary lines
            color="r",  # white color for centroids markings
            zorder=10,  # drawing order of axes
        )
        # describing text of the tuple will be title of the subplot
        plt.title(description)
        plt.xlim(xmin, xmax)  # limits of x-coordinates
        plt.ylim(ymin, ymax)  # limits of y-coordinates
        plt.xticks(())
        plt.yticks(())
        # Upper title of the whole plot
    plt.suptitle(
        # Text to be displayed
        "K-Medoids algorithm implemented with different metrics\n\n",
        fontsize=20,  # size of the fonts
    )

    input("Press To Continue")
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    create_graph()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
