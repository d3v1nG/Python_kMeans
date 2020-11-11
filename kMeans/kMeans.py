import random
import time
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# https://towardsdatascience.com/a-continuously-updating-k-means-algorithm-89584ca7ee63

class kMeans:
    def __init__(self, data, k, threshhold=0.00001, max_iter=300, shuffle=False, show_status=False):
        self.k = k
        self.threshhold = threshhold
        self.max_iter = max_iter
        self.data = data
        self.shuffle = shuffle
        self.show_status = show_status

        self.final_centroids = {}
        self.final_clusters = {}

        self.iter = 0

    def PlotPreview(self):
        plt.scatter(self.data[:,0], self.data[:,1], s = 15)
        plt.show()

    def ShowCentroids(self):
        print(self.final_centroids)

    def ShowClusters(self):
        for i in range(self.k):
            # mean = np.average(self.final_clusters[i],axis=0)
            mean = self.final_centroids[i]
            output = "[~] Cluster: "+ str(i) +", Cluster Size: "+str(len(self.final_clusters[i]))+ ", Mean: "+ str(mean)
            output+= "; Cluster Contents: "
            for point in self.final_clusters[i]:
                output += " ,"+str(point)
            print(output)
        print()


    def GenerateClusters(self):
        try:
            centroids = {}

            random.seed(420)
            for i in range(self.k):
                centroids[i] = self.data[random.randrange(len(self.data))]
                # centroids[i] = self.data[i]

            for iter in range(self.max_iter):
                clusters = {}
                # generate k amount of classification
                for i in range(self.k):
                    clusters[i] = []

                # added in case results are consitently wrong
                if self.shuffle:
                    np.random.shuffle(self.data)

    # get distances to centroids
                for point in self.data:
                    # default r param is 2, i used linalg.norm for eulicean distance
                    distances = [np.linalg.norm(point-centroids[centroid]) for centroid in centroids]
                    classification = distances.index(min(distances))
                    clusters[classification].append(point)

                original_centroids = dict(centroids)

                #Find Mean for each cluster
                for classification in clusters:
                    mean = np.average(clusters[classification],axis=0)
                    centroids[classification] = mean

                # lol, lack of a better word
                optimized = True

                # calculate Delta, the sum of the squared distance between the old means and the new means
                for c in centroids:
                    original_centroid = original_centroids[c]
                    current_centroid = centroids[c]
                    delta = np.sum(np.linalg.norm(current_centroid - original_centroid))
                    if delta > self.threshhold:
                        optimized = False
                print("Iteration : "+ str(iter))
                print("Current delta : "+ str(delta))

                if optimized:
                    # print("Final Delta : " + str(delta))
                    print("[+] Total iterations : "+ str(iter+1))
                    print("[+] Final Delta : "+ str(delta)+ "")
                    self.iter = iter
                    self.final_clusters = dict(clusters)
                    self.final_centroids = dict(centroids)
                    break

                # added for the sake of seeing each iteration's plot
                if (self.show_status):
                    colors = 10*["g","r","c","k", "m", "pink", "aqua"]
                    # colors = array(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
                    for classification in clusters:
                        color = colors[classification]
                        for p in clusters[classification]:
                            plt.scatter(p[0], p[1], marker="x", color=color, s=10, linewidths=1)
                    # place centroid on 'top' layer
                    for mean in centroids:
                        plt.scatter(centroids[mean][0], centroids[mean][1],
                                    marker="o", s=150, color="w", linewidths=1, edgecolors="b")
                    for mean in original_centroids:
                        plt.scatter(original_centroids[mean][0], original_centroids[mean][1],
                                    marker="s", s=30, color="w", linewidths=1, edgecolors="b")
                    plt.show()
        except KeyboardInterrupt:
            print("[-] KeyboardInterrupted, DONE")



    def PredictClassification(self, nData):
        distances = [np.linalg.norm(nData-self.final_centroids[centroid]) for centroid in self.final_centroids]
        classification = distances.index(min(distances))
        return classification

    def PlotClusters(self):
        colors = 10*["g","r","c","k", "m","y", "pink", "aqua"]
        # colors = array(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        for classification in self.final_clusters:
            color = colors[classification]
            for p in self.final_clusters[classification]:
                plt.scatter(p[0], p[1], marker="x", color=color, s=5, linewidths=1)
        # place centroid on 'top' layer
        for c in self.final_centroids:
            plt.scatter(self.final_centroids[c][0], self.final_centroids[c][1],
                        marker="o", color="w", s=130, linewidths=1, edgecolors="b")
        plt.show()
