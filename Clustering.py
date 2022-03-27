import sklearn
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansClass:
    """  performs k-means clustering  """
    _plt = None

    def __init__(self, k=4):
        self.k = k          # number of clusters
        self.means = None   # means of clusters
        self._plt = plt

    def Calculate_Silhoutte_Score(self, X , _km , _metric='euclidean'):
        score = silhouette_score(X, _km.labels_, metric=_metric)
        # Print the score
        return  score

    def inertia_vs_k_plot(self, input , K_range=10):
        kmeans_per_k = [KMeans( n_clusters = k, random_state=42).fit(input) for k in range(1, K_range)]   #init='k-means++' ,
        inertias = [model.inertia_ for model in kmeans_per_k]

        print("inertias : ",inertias)

        plt.figure(figsize=(8, 3.5))
        plt.plot(range(1, K_range), inertias, "bo-")
        #plt.text(0.5, 0.5, 'inertia vs k',transform=plt,  fontweight='bold', va='top')
        plt.xlabel("$k$", fontsize=14)
        plt.ylabel("Inertia", fontsize=14)
        '''
        plt.annotate('Elbow',
                    xy=(4, inertias[3]),
                    xytext=(0.55, 0.55),
                    textcoords='figure fraction',
                    fontsize=16,
                    arrowprops=dict(facecolor='black', shrink=0.1)
                    )
                    '''
        plt.axis([1, K_range + 1, 0, max(inertias)*1.2])
        plt.show()

    def train(self, inputs):
        self.means = KMeans(n_clusters=self.k,n_jobs=-1, random_state=42)
        estimator = self.means.fit(inputs)
        print("Cluster Means Shape : ", self.means.cluster_centers_.shape)
        return estimator
    
    def plot_clusters(self, X, y=None , feature1 =0, feature2=1):
        self._plt.scatter(X[:, feature1], X[:, feature2], c=y, s=1)
        self._plt.xlabel("$f_1$", fontsize=14)
        self._plt.ylabel("$f_2$", fontsize=14, rotation=0)

    
     
    def plot_centroids(self,Cluster, circle_color='w', cross_color='k', feature1 =0, feature2=1):
        for C in range(0,Cluster.cluster_centers_.shape[0]):
            C_X=Cluster.cluster_centers_[C][feature1]
            C_Y=Cluster.cluster_centers_[C][feature2]
            print("cluster {} = {} , {}".format(C,C_X , C_Y))
            self._plt.scatter(C_X, C_Y, marker='o', s=35, linewidths=8, color=circle_color, zorder=10, alpha=0.9)
            self._plt.scatter(C_X, C_Y, marker='x', s=2, linewidths=12, color=cross_color, zorder=11, alpha=1)

    def plot_decision_boundaries(self,clusterer, X, resolution=1000, show_centroids=True,
                                show_xlabels=True, show_ylabels=True):
        mins = X.min(axis=0) - 0.1
        maxs = X.max(axis=0) + 0.1
        xx, yy = np.meshgrid(np.linspace(mins, maxs, resolution),
                            np.linspace(mins, maxs, resolution))
        Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(Z, extent=(mins, maxs, mins, maxs),
                    cmap="Pastel2")
        plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                    linewidths=1, colors='k')
        plot_data(X)
        if show_centroids:
            plot_centroids(clusterer.cluster_centers_)

        if show_xlabels:
            plt.xlabel("$x_1$", fontsize=14)
        else:
            plt.tick_params(labelbottom=False)
        if show_ylabels:
            plt.ylabel("$x_2$", fontsize=14, rotation=0)
        else:
            plt.tick_params(labelleft=False)
    
    def plot_data(self,X):
        plt.figure(figsize=(8, 4))
        plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
        plt.show()



    def plot_centroids_onAxis(self,Cluster, circle_color='w', cross_color='k', feature1 =0, feature2=1 , axis=None):
        for C in range(0,Cluster.cluster_centers_.shape[0]):
            C_X=Cluster.cluster_centers_[C][feature1]
            C_Y=Cluster.cluster_centers_[C][feature2]
            axis.scatter(C_X, C_Y, marker='o', s=35, linewidths=8, color=circle_color, zorder=10, alpha=0.9)
            axis.scatter(C_X, C_Y, marker='x', s=2, linewidths=12, color=cross_color, zorder=11, alpha=1)
                
    def plot_Box_Features_Centroids(self,Cluster, X):
        plt.rcParams["figure.figsize"] = [3*X.shape[1], 2*X.shape[1]]
        plt.rcParams["figure.autolayout"] = True
        f1 = X.shape[1]
        f2 = X.shape[1]
        fix, axes = plt.subplots(nrows=f1, ncols=f2)

        for i in range(f1):
            for j in range(f2):
                axes[i][j].scatter(X[:, i], X[:, j], c=None, s=1)
                self.plot_centroids_onAxis(Cluster.means, axis=axes[i][j], feature1=i,feature2=j)

        plt.show()




