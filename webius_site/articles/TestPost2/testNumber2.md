### Number 2
Let's now apply the SVM to some actual data. I will use the popular *Iris* data set, which contains features and labels of three distinct species of iris flowers. We use the scikit *LabelEncoder* to map the *species* label to a numerical category. However the *LabelEncoder* will by default return the labels 0 and 1, where 0 means not part of that species and 1 means is of species. Due to eq. 4 we require -1 and 1 as labels, so we have to apply our custom mapping.

```python
def load_data(cols):
    iris = sns.load_dataset("iris")
    iris = iris.tail(100)
 
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris["species"])
    
     # Map default labels (0, 1) to custom labels (-1, +1)
    custom_mapping = {0: -1, 1: +1}
    y = np.array([custom_mapping[label] for label in y])
    
    # drop the species column since it's now encoded in the labels y
    X = iris.drop(["species"], axis=1)
    
    # only consider the columns given:
    if len(cols) > 0:
        X = X[cols]
 
    return X.values, y
```
For simplicity and easier plotting we will also only consider two features of each flower. Here I have choosen the features *petal length* and *petal width*. For the curious: The petal is some specific leaf of iris flowers:\
![Petal](petal.webp "Image of an iris flower.")\
\
Finally let's write the driver code to load the data and feed it through the SVM classifier.

$$
\begin{align}
\frac{\partial L}{\partial w} = w - C \cdot \Sigma_{i=1,\xi_i>0}^N y_i x_i \\

\frac{\partial L}{\partial b} = C \cdot \Sigma_{i=1, \xi_i>0}^N y_i
\end{align}
$$