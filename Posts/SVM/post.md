# Building a Support Vector Machine from Scratch
Support Vector Machines (SVM) are a type of supervised machine learning algorithm used primarily for classification tasks.
The core idea behind SVM is to find the best boundary (or hyperplane) that separates a given set of data points, thus performing
the classification. In general we are interested in finding a hyperplane, which separates the different classes with the maximum possible
margin, that is the distance between the closest data points (called support vectors) from each class to the boundary.
In the first part of this post we will only consider linear-separabable problems, that is problems for which a straight hyperplane can
divide the classes. While this may sound very restrictive, note that we are not constrained to the features in the original data, but can
add new derived features. So while a feature vector with n dimensions (so n distinct features) may not be separable by an n-1 dimensional hyperplane,
we can by means of feature engineering construct new features which we add to each vector. Thus each vector becomes m-dimensional (m > n) effectively
mapping it into a high dimensional space. In this higher dimensional space a hyperplane which cleanly separates the data may exist.

The hyperplane is a decision boundary in an n-dimensional space. Any hyperplane (for our n-dimensional data vectors $\mathbf{x_i}$) can be described using the following
equation:
$$
\begin{equation}
\mathbf{w} \cdot \mathbf{x} - b = 0
\end{equation}
$$
where $\mathbf{w}$ is a real valued vector of the same dimension as our data vectors $x$ and b is a real number. So each $\mathbf{x_i}$ has n features (dimensions) and there are N of them, each represents a single example and has a label $y_i$. Suppose we found an appropriate hyperplane for the classification taks at hand. We could then predict the label (i.e. the class) of some new input feature vector $x$ by evaluating:
$$
\begin{equation}
y = sign(\mathbf{wx}-b)
\end{equation}
$$
Where sign simply returns +1 if $wx-b > 0$ and -1 if $wx-b < 0$. So effectively eq. tells us on which side of the hyperplane the new feature vector resides and returns the corresponding label.
The goal of SVM is to leverage our dataset to find the optimal values $w*$ and $b*$, which form a beneficial hyperplane. Once that hyperplane is found
eq. 2 defines our model.
We can find $w*$ and $b*$ by solving an optimization problem. The optimization problem we have to solve is finding the hyperplane with the largest margin under certain constraints. Note that SVM is a supervised learning algorithm, which is trained on example data. Each example consists of some feature vector $x_i$ and it's corresponding
label $y_i$: $(x_i, y_i)$. Where $x_i$ is a column vector with n features. Hence we can require, that:
$$
\begin{equation}
\begin{cases}
wx_i - b >= +1 & \text{if } y_i = +1\\
wx_i - b <= -1 & \text{if } y_i = -1
\end{cases}
\end{equation}
$$
This allows us to combine the two constraints into a single equation:
$$
\begin{equation}
(\mathbf{wx_i} - b) \cdot y_i >= 1
\end{equation}
$$
The margin of the hyperplane (which we want to maximize) is given by:
$margin = \frac{2}{||w||}$

In summary we can thus say that we want to minimize $||w||$ while respecting $(\mathbf{wx_i} - b) \cdot y_i >= 1$, $i$ from 1..N.
In practice we will not minimize $||w||$ but 
$$
\begin{equation}
L = min \left( \frac{1}{2} ||w||^2 \right)
\end{equation}
$$
which is equivalent, but does allow us to perform quadratic optimizations later on. Eq. 5 represents our primal loss function which we need to minimize in order to find a good hyperplane. It is the loss function for the hard margin classifier.

### Dealing with noise
As you know real world data sets are far from perfect and exhibit missing data points and noise. So how can we deal with misclassifications introduced due to noise in our data and still obtain a good classifier? In order to tolerate misclassifications up to a certain point, we expand eq. 5 by introducing slack variables $\xi_i$. These
slack variables soften the inequality constraint by allowing certain feature vectors to violate the margin.
$$
\begin{equation}
y_i (\mathbf{w} \mathbf{x_i} - b) >= 1 - \xi_i
\end{equation}
$$
For $0 < \xi_i < 1$ the point is within the margin and classified correctly, but the margin is smaller than we like it to be. For $\xi >= 1$ the point is misclassified and resides on the wrong side of the hyperplane. So in general we want these slack variables to stay as small as possible. We can achieve this by adding them to our loss function (see eq. 5):
$$
\begin{equation}
min \left( \frac{1}{2} ||w||^2 + C \cdot \Sigma_{i=1}^N \xi_i^k \right)
\end{equation}
$$
Here we also introduced the hyperparamter C, which balances the cost of misclassifications. For $C=0$ the algorithm will only maximize the margin, for $C=\infty$ the size of the margin is neglected and the algorithm will just try to minize the loss. So C controls the trade-off between maximizing the margin and minimizing the loss. $k$ is another hyperparameter which controls the impact of the slack variables and is typically selected to be 1 or 2. Since we now allow some data points to fall within the margin or even the wrong side of the hyperplane, this is the loss used in a soft margin classifier (in contrast to the hard margin classifier).\
We would like to include the constraint given by eq. 6 into our loss function so we can then solve it using gradient descend. So let's solve eq. 6 for $\xi_i$
$$
\begin{equation}
\xi_i >= 1 - y_i (\mathbf{w} \mathbf{x_i} - b)
\end{equation}
$$
However gradient descend cannot handle inequalities such as the one above. So we cannot yet plug in that inequality into our loss function. To incorporate the slack variables into the loss function we have to tweak it a little. Let's consider the values $\xi_i$ can have and their respective meaning for the classifier:
$$
\begin{equation}
\begin{cases}
y_i (\mathbf{w x_i} + b) >= 1 & \text{then } \xi_i < 0 \text{ sample satisfies eq. 4}\\
y_i (\mathbf{w x_i} + b) < 1 & \text{then } \xi_i > 0 \text{ sample does not satisfy eq. 4, slack is required}
\end{cases}
\end{equation}
$$
These considerations motivate the hinge loss function, which simply becomes zero, when the constraint given by eq. 4 is satisified; that is if $\mathbf{wx_i}$ lies on the correct side of the decision boundary. For data which lies on the wrong side of the boundary the hinge loss function produces a penalty value, which is proportional to that data's offset from the decision boundary.
$$
\begin{equation}
\xi_i' = max \left(0,\ 1 - y_i (\mathbf{wx_i} - b) \right)
\end{equation}
$$
Hinge loss allows us to consider noisy data, which may not always reside on the proper side of the hyperplane. Another nice property of the hinge loss function is that it's subdifferentiable, that is a gradient can be computed on each segment of the piecewise function. We can now incorporated hinge loss into our previous loss function (see eq. 7). When we replace the slack variables $\xi_i$ with the hinge loss function and choose $k=1$ we obtain:
$$
\begin{equation}
L = min \left( \frac{1}{2} ||w||^2 + C \cdot \Sigma_{i=1}^N max \left(0,\ 1 - y_i (\mathbf{wx_i} - b) \right) \right)
\end{equation}
$$
Note how this formulation allowed us to encode the contraints given by eq. 4 into our loss function. Thus we are no longer concerned with two simultaneous problems, but can no focus on minimizing $L$ and thereby solve the optimization problem. The first part of our loss function is fully differentiable and since the hinge loss function is subdifferentiable the combined loss function is now suitable for gradient descend. This is simply done by computing the partial derivatives of L with respect to $w$ and $b$, because these are the parameters we want to find.
$$
\begin{align}
\frac{\partial L}{\partial w} = w - C \cdot \Sigma_{i=1,\xi_i>0}^N y_i x_i \\

\frac{\partial L}{\partial b} = C \cdot \Sigma_{i=1, \xi_i>0}^N y_i
\end{align}
$$
Note that we can simply neglect all terms for which the hinge loss $\xi_i$ would be zero, because these terms are themselves zero.

### Implementation of the linear SVM classifier with gradient descend

```python
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

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
```

```python
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
```

```python
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
```