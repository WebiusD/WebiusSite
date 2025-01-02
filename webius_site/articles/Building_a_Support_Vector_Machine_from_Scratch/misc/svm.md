# Support vector machines
- in order to separate data it helps to transform them into higher dimensional spaces
- this mapping is done by adding additional features.
- these features can be found out -> feature engineering
- in this higher dimensional space the data then becomes separable
- however explicitly transforming the original data (feature vectors) into the higher dimensional space will be very costly for high dimensions.

- for example
- (x1, x2) -> (1, x1, x2, x1*x2, x1^2, x2^2)

- the normal (primal) objective function which we have to solve is to find the coefficients for beta and the bias b which minimize beta^2 / 2
  under some constraint. See here: https://adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-linear-svm/
  By minimizing this objective function, we maximize the margin, which is the distance of the hyperplane to the datapoints, it should divide
- The minimization problem together with the conditions can be reformulated using Lagrangian multipliers, which results in a set of equations we
  have to solve. See: https://adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-duality-problem/
- The lagrangian is then a function of beta and b and the multipliers alpha. The lagrangian is only a useful way to solve the original problem,
  but under certain constraints.
- Since we know the solution of the lagrangian has to be stationary w.r.t beta, b and alpha, we look for a solution of the equation which satisfy the
  stationarity conditions.
- After some evaluation, we find that the final equation only depends on the inner product of the data vectors in the input space: x_i * x_j for all pairs i, j

- Since there is a well known and cheap way to transform the inner product of vectors into a higher dimensional space, we can now replace the inner
  product of feature vectors with the inner product of the transformed feature vectors, thus effectively solving the the problem in a higher dimensional space
  where the separation of the data is possible. The function which allows us to convert the inner product of our original vectors to the inner product of 
  transformed vectors is called a kernel function.
- Through kernel functions we can avoid to explicitly transform the original data (feature vectors) into the higher dimensional space, which is costly.
- Kernel functions are very cheap to compute, since they do not explictly transform the data, but just tell us how the inner product of the data
  changes when doing the transformation.

# So now we have all parts together to solve the minimize the objective function. Summarized:
1) Find the dual problem of the original problem, that is finding a max. margin hyperplane which separates the samples into categories by solving the objective function.
2) Formulate the dual problem, which will only be a function of the inner product of the data vectors in the input space.
3) Rewrite the dual problem but now in the feature space, where the separation is possible. Simply replace x_i and x_j with T(x_i) and T(x_j) respectively.
4) Use a kernel to compute how that inner product changes when going into high dimensional space. Or better, replace T(x_i) * T(x_j) with K(x_i, x_j) 
in the dual objective function.
4) Since the dual objective function needs to be maximized, use gradient ascend (not descend) in order to find a good solution.