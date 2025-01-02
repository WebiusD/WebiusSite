A list of vectors in a vector space that is small enough to be linearly independent
and big enough so the linear combinations of the list fill up the vector space is
called a basis of the vector space.

<h3>The standard basis</h3>
When we talk about vectors we, mostly implicitly assume the standard basis, so in a carthesian vector system:
$\newcommand\cvec[1]{\begin{bmatrix}#1\end{bmatrix}}$

$$v = \cvec{a\\b\\c}$$
stands for:
$$a \cdot e_1 + b \cdot e_2 + c \cdot e_3\tag{1}$$
where $e_1, e_2$ and $e_3$ represent the unit vectors along the carthesian axes and form the basis for a three dimensional vector space. While it is easiest for us to think in terms of the standard basis, it is by no means the only basis that spans this vector space. A list of vectors form the basis of a vector space, if it satisfies two conditions:
<ul>
<li>The basis vectors are linear independent</li>
<li>The basis vectors span the entire vector space</li>
</ul>
The latter condition is just the mathematical way of saying that the basis vectors can be combined (through addition and scalar multiplication) to represent any vector within that space. The scalars we write down, when we refer to a certain vector are called the vectors components, and as evident from equation (1), are just the values each basis vector is multiplied by. A basis can be understood as an ordered list of basis vectors. So the standard basis could simply be written down as:
$$E = [e_1, e_2, e_3]$$
Note that the order of the basis vectors is important, since we must know which component is associated with which basis vector. So one way to represent our vector is with the components a, b and c in the standard basis. However the same vector could be represented in another basis - resulting in different components. We will see how to change to a new basis shortly. In the following we will deal with multiple basis, so let's introduce the following notation:
$$v = a \cdot e_1 + b \cdot e_2 + c \cdot e_3 = \cvec{a\\b\\c} = [v]_E$$
That is, the component vector $v$ relative to the Standardbasis $E$. In the following this convention that each component has to be multiplied with its respective basis vector will prove to be very useful.
<br>
<br>
<br>

<h3>Changing the basis - the general case</h3>
Suppose we have a vector $v$ expressed with respect to some arbitrary basis $U={u_1, u2, u3}$, i.e. $[v]_U$. Let the components of $v$ be, $c_1, c_2$ and $c_3$:
$$[v]_U = c_1 \cdot u_1 + c_2 \cdot u_2 + c_3 \cdot u_3\tag{2}$$
What are the components of $v$ in another basis $W$? So how do we go from $[v]_U$ to $[v]_W$?. As it turns out we simply have to express the basis vectors of $U$, using the basis vectors of $W$. In other words we have to solve the following equations:
$$
[u_1]_W = a_{11} \cdot w_1 + a_{21} \cdot w_2 + a_{31} \cdot w_3\\
[u_2]_W = a_{12} \cdot w_1 + a_{22} \cdot w_2 + a_{32} \cdot w_3\\
[u_3]_W = a_{13} \cdot w_1 + a_{23} \cdot w_2 + a_{33} \cdot w_3\tag{3}
$$
And here is why. Starting out with the definition of $[v]_U$ and then substituting equation (3):
$$
v = [v]_U = c_1 \cdot u_1 + c_2 \cdot u_2 + c_3 \cdot u_3\\
= c_1 \cdot \left(a_{11} \cdot w_1 + a_{21} \cdot w_2 + a_{31} \cdot w_3\right)\\
+ c_2 \cdot \left(a_{12} \cdot w_1 + a_{22} \cdot w_2 + a_{32} \cdot w_3\right)\\
+ c_3 \cdot \left(a_{13} \cdot w_1 + a_{23} \cdot w_2 + a_{33} \cdot w_3\right)\\
\\
= \left(c_1 \cdot a_{11} + c_2 \cdot a_{12} + c_3 \cdot a_{13}\right) \cdot w_1\\
+ \left(c_1 \cdot a_{21} + c_2 \cdot a_{22} + c_3 \cdot a_{23}\right) \cdot w_2\\
+ \left(c_1 \cdot a_{31} + c_2 \cdot a_{32} + c_3 \cdot a_{33}\right) \cdot w_3\\
$$
By using our convention from before. This can be written as:
$$
[v]_W = \cvec{c_1 \cdot a_{11} + c_2 \cdot a_{12} + c_3 \cdot a_{13}\\c_1 \cdot a_{21} + c_2 \cdot a_{22} + c_3 \cdot a_{23}\\c_1 \cdot a_{31} + c_2 \cdot a_{32} + c_3 \cdot a_{33}}\\
= \begin{bmatrix}
a_{11}&a_{12}&a_{13}\\
a_{21}&a_{22}&a_{23}\\
a_{31}&a_{32}&a_{33}
\end{bmatrix} \cdot \cvec{c_1\\c_2\\c_3} = T_{U \rightarrow W} \cdot [v]_U
$$
Here we defined the matrix which takes us from basis U to basis W as $T_{U \rightarrow W}$. When we compare this to equation (3) we see that the coefficients $a_{ij}$ required to compose the basis vectors of $U$ in $W$ are laid out in the columns of $T_{U \rightarrow W}$:
$$
T_{U \rightarrow W} = \begin{bmatrix}[u_1]_W&[u_2]_W&[u_3]_W\end{bmatrix}\tag{4}
$$
This is the core insight of this article and a lot will follow from this in a very straight forward manner. 
<br>
<br>
<br>

<h3>An example in 2D</h3>
Lets consider a concrete example. For the sake of simplicity we will stick to two dimension. Say we have a vector $v = [v]_U = \cvec{1\\2}$ specified in the basis $U$ with basis vectors:
$$
u_1 = \cvec{2\\-1},\ 
u_2 = \cvec{1\\3}
$$ 
And we want to transform our vector $v$ to a new basis $W$ with basis vectors:
$$
w_1 = \cvec{1\\-2},\
w_2 = \cvec{0\\1} 
$$ 
In order to find the transformation matrix $T_{U \rightarrow W}$ we have to express the basis vectors of the start basis $U$ within the target basis $W$.
$$
T_{U \rightarrow W} = \begin{bmatrix}[u_1]_W&[u_2]_W\end{bmatrix}
$$
So we want to solve:
$$
u_1 = a_{11} \cdot w_1 + a_{21} \cdot w_2\\
u_2 = a_{12} \cdot w_1 + a_{22} \cdot w_2
$$
For two dimensions this is relatively straigh forward. With some intuition we can see that:
$$
u_1 = 2 \cdot w_1 + 3 \cdot w_2\\
u_2 = 1 \cdot w_1 + 5 \cdot w_2
$$
Now we simply need to write these coefficients into the columns of a 2 x 2 matrix and get:
$$
[v]_W = T_{U \rightarrow W} \cdot [v]_U = \begin{bmatrix}2&1\\3&5\end{bmatrix} \cvec{1\\2} = \cvec{4\\13}
$$

<br>
<br>
<br>
<h3>Utilizing the standard basis</h3>
What could still be solved with a little trial and error in two dimensions becomes significantly more difficult as the dimension increases. In general, in $n$ dimensions, a system of equations must be solved with an equal number of linear equations. However, in this section, we will discover how to bring the problem into a nicer form, resulting in a much more pleasant calculation.
The idea is to perform the transformation from basis A to basis B not directly, but by using the standard basis $E$ as an intermediary:
$$
T_{A \rightarrow B} = T_{E \rightarrow B} \cdot T_{A \rightarrow E}
$$
The single complex transformation is subdivided into two transformations that have nice properties, as we will see shortly. At the beginning of this article we have introduced the standard basis, which is soo deeply rooted into our mathematical thinking, that we often aren't aware that we are using it. Observe what happens when we want to transform a vector defined with respect to some arbitrary complex basis $A$ to the standardbasis $E$. Once more we can make use of equation (4). We can find the new components simply by expressing the basis vectors of $A$ in the standardbasis and then laying them out into the columns of our transformation matrix $T_{A \rightarrow E}$. But now comes the clue the basis vectors of $A$ are already expressed in the standardbasis, because its the basis we use when we write down the basis vectors $a_1\ ...\ a_n$. Hence $T_{A \rightarrow E}$ simply becomes:
$$
T_{A \rightarrow E} = \begin{bmatrix}[a_1]_E&[a_2]_E&[a_3]_E\end{bmatrix} = \begin{bmatrix}a_1&a_2&a_3\end{bmatrix}
$$
Well and good, $T_{A \rightarrow E}$ is simple, but what about $T_{E \rightarrow B}$? It's also simple! Because we know, that for any transformation matrix and any bases $U$, $W$ it must hold:
$$
T_{U \rightarrow W} \cdot T_{W \rightarrow U} = I
$$
and hence:
$$
T_{U \rightarrow W} = \left(T_{W \rightarrow U}\right)^{-1}
$$
and vice versa. So instead of $T_{E \rightarrow B}$ we simply use $T_{B \rightarrow E}$ and invert it. With this we have found an easy to construct formula for $T_{A \rightarrow B}$:
$$
T_{A \rightarrow B} = T_{E \rightarrow B} \cdot T_{A \rightarrow E} = \left(T_{B \rightarrow E}\right)^{-1} \cdot T_{A \rightarrow E} = \left(\begin{bmatrix}b_1&b_2&b_3\end{bmatrix}\right)^{-1} \cdot \begin{bmatrix}a_1&a_2&a_3\end{bmatrix} 
$$
Let's leverage this formula to validate the results of our earlier 2d example. All we have to do is write the basis vectors of $U$ and $W$ into the columns of the two matrices and invert the latter. For those who forgot inverting a 2d matrix comes down to swapping the entries on the diagonal, inverting the sign of the entries on the off-diagonal and finally divide everything by the determinant (which is 1 in our case).
$$
T_{U \rightarrow W} = \left(\begin{bmatrix}1&0\\-2&1\end{bmatrix}\right)^{-1} \cdot \begin{bmatrix}2&1\\-1&3\end{bmatrix} = \begin{bmatrix}1&0\\2&1\end{bmatrix} \cdot \begin{bmatrix}2&1\\-1&3\end{bmatrix} = \begin{bmatrix}2&1\\3&5\end{bmatrix}
$$
This results in a transformed vector $v$:
$$
[v]_W = \begin{bmatrix}2&1\\3&5\end{bmatrix} \cdot \cvec{1\\2} = \cvec{4\\13}
$$
reproducing our previous result.
<br>
<br>
<br>