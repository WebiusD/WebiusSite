import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

from code import interact

def load_data(cols):
    iris = sns.load_dataset("iris")
    iris = iris.tail(100)
 
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris["species"])
    # use -1 label instead of 0, the other label stays +1
    y[y == 0] = -1
    
    # drop the species column since it's now encoded in the labels y
    X = iris.drop(["species"], axis=1)
    
    # only consider the columns given:
    if len(cols) > 0:
        X = X[cols]
 
    return X.values, y


class LinearSVM:
    def __init__(self, C=1.0):
        self._support_vectors = None
        self.C = C
        self.w = None
        self.b = 0
        self.X = None
        self.y = None

        # number of data points
        self.n_data = 0

        # number of dimensions:
        self.n_dim = 0
    
    def __decision_fn(self, X):
        return X.dot(self.w) - self.b
    
    def __margin(self, X, y):
        return y * self.__decision_fn(X)
    
    def __cost(self, margin):
        return (1/2)* self.w.dot(self.w) + self.C * np.sum(np.maximum(0, 1-margin))
    
    def fit(self, X, y, lr=1e-3, epochs=500):
        # for plotting:
        self.X = X
        self.y = y
        #-------------

        self.n_data, self.n_dim = X.shape   # (100, 2) in our case
        self.w = np.random.randn(self.n_dim)
        self.b = 0

        losses = []
        for _ in range(epochs):
            margin = self.__margin(X, y)
            loss = self.__cost(margin)
            losses.append(loss)

            # compute the misclassified points, because these are the only ones where
            # the hinge loss is non-zero. The misclassified points are those for which:
            # y (wx + b) < 1, so margin < 1:
            misclassified_idx = np.where(margin < 1)[0]

            # calculate the derivates (gradient)
            dL_dw = self.w - self.C * y[misclassified_idx].dot(X[misclassified_idx])
            dL_db = self.C * np.sum(y[misclassified_idx])

            # update:
            self.w -= lr* dL_dw
            self.b -= lr* dL_db

        self._support_vectors = np.where(self.__margin(X, y) <= 1)[0]

    def predict(self, X):
        return np.sign(self.__decision_fn(X))
    
    def score(self, X, y):
        P = self.predict(X)
        return np.mean(y == P)
    
    def plot_decision_boundary(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=50, cmap=plt.cm.Paired, alpha=.7)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.__decision_fn(xy).reshape(XX.shape)

        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors=['r', 'b', 'r'], levels=[-1, 0, 1], alpha=0.5,
                    linestyles=['--', '-', '--'], linewidths=[2.0, 2.0, 2.0])

        # highlight the support vectors
        ax.scatter(self.X[:, 0][self._support_vectors], self.X[:, 1][self._support_vectors], s=100,
                    linewidth=1, facecolors='none', edgecolors='k')

        plt.show()
    

if __name__ == "__main__":
    # only consider petal_length and petal_width:
    cols = ["petal_length", "petal_width"]
    X, y = load_data(cols)

    # scale the data:
    X = StandardScaler().fit_transform(X)

    model = LinearSVM(C=15.0)
    model.fit(X, y)
    print(f"Train score: {model.score(X, y)}")

    model.plot_decision_boundary()