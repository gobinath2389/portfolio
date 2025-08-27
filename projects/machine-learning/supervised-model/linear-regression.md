# Linear Regression model

The linear regression hypothesis function can be represented as:

$$
h(x) = \theta^T x
$$

where:

- $\theta$ and $x$ are vectors  
- $\theta$ is the set of parameters (weights)  
- $x$ is the input feature values  

$\theta^Tx$ -> The dot product of parameters and features.

$$
h(x) = \theta^T x = \theta_0 \cdot 1 + \theta_1 x_1 + \theta_2 x_2
$$


The vectors look like this:

$$
\theta =
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\theta_2
\end{bmatrix},
\quad
x =
\begin{bmatrix}
1 \\
x_1 \\
x_2
\end{bmatrix}
$$


Suppose you want to predict house prices (y) based on:

$x_1$: size in square feet

$x_2$: number of bedrooms
\section*{Linear Regression Model Example}

The hypothesis function can be written as:

$$
h(x) = \theta_0 + \theta_1 \cdot (\text{size}) + \theta_2 \cdot (\text{bedrooms})
$$

Where:

- If $\theta_1 = 200$, it means each additional square foot adds \$200 to the price.
- If $\theta_2 = 10{,}000$, it means each extra bedroom adds \$10,000.
- If $\theta_0 = 50{,}000$, it means the base price of the house (even with 0 size and 0 bedrooms) is \$50,000.  


<br/>
Now, given a training set, how do we pick, or learn, the parameters θ?
One reasonable method seems to be to make h(x) close to y, at least for
the training examples we have. To formalize this, we will define a function
that measures, for each value of the θ’s, how close the h(x
(i)
)’s are to the
corresponding y
(i)
’s. We define the cost function:  

<br/>


 $J(\theta) = \frac{1}{2} \sum_{i=1}^{n} \big(h_\theta(x^{(i)}) - y^{(i)}\big)^2$
 
<br/>

# Stochastic Gradient Descent vs Batch Gradient Descent

## 1. Batch Gradient Descent (BGD)
In **batch gradient descent**, for every update of parameters $\theta$, you use **all training examples**.  

Formula:

$$
\theta_j := \theta_j + \alpha \cdot \frac{1}{m} \sum_{i=1}^m \Big( y^{(i)} - h_\theta(x^{(i)}) \Big) x^{(i)}_j
$$

- The gradient is computed using the **entire dataset**.  
- Only **one update per epoch** (a full pass over the training set).  
- Computationally expensive when $m$ (number of samples) is very large.  

---

## 2. Stochastic Gradient Descent (SGD)
In **SGD**, you update $\theta$ **after every single training example**.  

Formula:

$$
\theta_j := \theta_j + \alpha \Big( y^{(i)} - h_\theta(x^{(i)}) \Big) x^{(i)}_j
$$

- As soon as you see one training point $(x^{(i)}, y^{(i)})$, you immediately update $\theta$.  
- This is why it’s called **incremental update**: parameters are adjusted step by step, one training sample at a time.  

---

## 3. Why is this useful?
- **Faster progress early** → You don’t need to wait for a whole dataset pass.  
- **Can handle large datasets** → Works even when data doesn’t fit in memory (processes one point at a time).  
- **Adds randomness (noise)** → Helps avoid getting stuck in local minima and can improve generalization.  

---

## 4. Intuition
Think of training like **walking downhill** on a landscape:

- **Batch Gradient Descent** → Look at the entire mountain first, compute the exact best direction, then take one step.  
- **SGD** → Look at just the ground under your foot (one sample), take a step based on that local slope. You might wobble, but you’ll keep moving faster.  

---

# normal equation
The normal equation is a mathematical method used in linear regression to find the optimal paramters $\theta$ that minimize the cost function $J(\theta)$ without using iterative methods like gradient decent. it Gives a closed form solution.

## formula
For a linear regression model:
        $h_\theta(x)=X\theta$ 

The Normal Equation computes:
    $\theta=(X^TX)^{-1} X^Ty$

Where:  

- $X$ = matrix of input features (with a column of 1’s for the intercept)  
- $y$ = vector of output values  
- $\theta$ = vector of parameters (weights)  
- $X^T$ = transpose of $X$  
- $(X^T X)^{-1}$ = inverse of $X^T X$  

## Key Points

1. No need to choose a learning rate (unlike gradient descent).

2. Works only if $(X^T X)$ is invertible

3. Computationally expensive for very large datasets because of the matrix inversion 𝑂(𝑛3)O(n3) complexity.

## Explanation for $(X^T X)^{-1}$ = inverse of $X^T X$  

# Normal Equation Example

---

### 1. Reminder: The Normal Equation

In linear regression, the solution for parameters is:

$$
\theta = (X^T X)^{-1} X^T y
$$

Here, $(X^T X)^{-1}$ means the **inverse of the matrix product** $X^T X$.

---

### 2. Example dataset

Suppose we have **2 training examples** and **1 feature (plus intercept)**.

Training data:

$$
(x, y) = \{(1, 1), (2, 2)\}
$$

---

### 3. Construct the design matrix $X$
The **design matrix** \(X\) is just a way of putting all your training data into a single matrix so that we can write the hypothesis in matrix form.

$$

X =
\begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & \dots & x_m^{(1)} \\
1 & x_1^{(2)} & x_2^{(2)} & \dots & x_m^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(n)} & x_2^{(n)} & \dots & x_m^{(n)}
\end{bmatrix}

$$

- **Each row** = one training example  
- **Each column** = one feature  
- The **first column of 1’s** = the intercept term (bias)

We add a column of 1’s for the intercept:

$$
X =
\begin{bmatrix}
1 & 1 \\
1 & 2
\end{bmatrix}, \quad
y =
\begin{bmatrix}
1 \\
2
\end{bmatrix}
$$

---

### 4. Compute $X^T X$

First, transpose $X$:

$$
X^T =
\begin{bmatrix}
1 & 1 \\
1 & 2
\end{bmatrix}
$$

Now multiply:

$$
X^T X =
\begin{bmatrix}
1 & 1 \\
1 & 2
\end{bmatrix}
\begin{bmatrix}
1 & 1 \\
1 & 2
\end{bmatrix}
=
\begin{bmatrix}
2 & 3 \\
3 & 5
\end{bmatrix}
$$

---

### 5. Compute the inverse $(X^T X)^{-1}$

For a $2 \times 2$ matrix:

$$
A =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}, \quad
A^{-1} = \frac{1}{ad - bc}
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
$$

For our case:

$$
X^T X =
\begin{bmatrix}
2 & 3 \\
3 & 5
\end{bmatrix}
$$

Determinant:

$$
\det = (2)(5) - (3)(3) = 10 - 9 = 1
$$

So:

$$
(X^T X)^{-1} =
\begin{bmatrix}
5 & -3 \\
-3 & 2
\end{bmatrix}
$$

---

### 6. Put it all together

Now the normal equation:

$$
\theta = (X^T X)^{-1} X^T y
$$

Compute $X^T y$:

$$
X^T y =
\begin{bmatrix}
1 & 1 \\
1 & 2
\end{bmatrix}
\begin{bmatrix}
1 \\
2
\end{bmatrix}
=
\begin{bmatrix}
3 \\
5
\end{bmatrix}
$$

Now multiply:

$$
\theta =
\begin{bmatrix}
5 & -3 \\
-3 & 2
\end{bmatrix}
\begin{bmatrix}
3 \\
5
\end{bmatrix}
=
\begin{bmatrix}
0 \\
1
\end{bmatrix}
$$

---

### 7. Interpretation

So the fitted model is:

$$
h_\theta(x) = 0 + 1 \cdot x
$$

Which is just:

$$
h(x) = x
$$

And that matches our training data perfectly:

- If $x = 1$, prediction = 1  
- If $x = 2$, prediction = 2  

So in this example, $(X^T X)^{-1}$ was the **key step** that let us solve for $\theta$ without gradient descent.


# Why Do We Take Derivatives to Minimize a Function?

## 1. What does it mean to “minimize”?
Suppose we have a function (like a cost function in machine learning):

$$J(\theta) = (\theta - 2)^2$$

- The **minimum** is the lowest point (bottom of the U-shape).  
- We want to **find the value of \(\theta\)** that gives this minimum.

---

## 2. How derivatives help
The **derivative** of a function tells us its **slope** (rate of change).  

- If slope \(> 0\) → the function is increasing.  
- If slope \(< 0\) → the function is decreasing.  
- If slope \(= 0\) → the function is flat → this could be a **minimum**, a **maximum**, or a **saddle point**.

 So, to find the minimum, we solve:



$$\frac{d}{d\theta} J(\theta) = 0$$

---

## 3. Example with steps


$$J(\theta) = (\theta - 2)^2$$

**Step 1: Take derivative**

$$
\frac{d}{d\theta} J(\theta) = 2(\theta - 2)$$


**Step 2: Set derivative = 0**

$$2(\theta - 2) = 0 \quad \Rightarrow \quad \theta = 2$$



**Step 3: Confirm it’s a minimum**

Second derivative test:



$$\frac{d^2}{d\theta^2} J(\theta) = 2 > 0$$

So the minimum happens at \(\theta = 2\).

---

## 4. Geometric intuition
Imagine you’re hiking on a mountain ⛰️:

- Walking **downhill** = slope negative  
- Walking **uphill** = slope positive  
- At the **valley bottom** = slope zero  

That "slope zero" point is exactly why we use **derivatives**.

---

## 5. Why in Machine Learning?
Our **cost function** \(J(\theta)\) is usually something like:

$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2
$$

We want to **minimize** this cost → find the best parameters \theta
So, we take derivatives of $$J(\theta)$$ w.r.t. each parameter \(\theta_j\), set them to 0, or use **gradient descent**.

---

## ✅ Summary
We take derivatives to find where the slope is zero → candidate minimums.  
In optimization (like ML training), this lets us find the **best parameters** that minimize the cost.
