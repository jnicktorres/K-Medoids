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


def create_graph():

    ds = load_digits()
    dig_data = scale(ds.data)
    num = len(np.unique(ds.target))
    red = PCA(n_components=2).fit_transform(dig_data)

    h = 0.02
    xmin, xmax = red[:, 0].min() - 1, red[:,0].max() + 1
    ymin, ymax = red[:, 1].min() - 1, red[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))

    models = [
        (
            KMedoids(metric="manhattan", n_clusters=3,
                     init="heuristic", max_iter=2), "Manhattan metric",
        ),
        (
            KMedoids(metric="euclidean", n_clusters=3,
                     init="heuristic", max_iter=2), "Euclidean metric",
        ),
        (KMedoids(metric="cosine", n_clusters=3, init="heuristic",
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
        model.fit(red)
        # Predict the labels for points in the mesh
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
            red[:, 0], red[:, 1], "k.", markersize=2, alpha=0.3
        )
        # Plot the centroids as white cross marks
        centroids = model.cluster_centers_
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,  # marker’s size (points^2)
            linewidths=3,  # width of boundary lines
            color="w",  # white color for centroids markings
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
    plt.show()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    create_graph()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
