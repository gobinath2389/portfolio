# Probabilistic Interpretations
A probabilistic interpretation means that instead of treating a model as a black-box function that just spits out predictions, we view it as describing a probability distribution over possible outcomes.

So, predictions aren’t just numbers — they are connected to probabilities and uncertainty.

### Why is it useful?
    Uncertainty matters → A doctor, a self-driving car, or a fraud detector doesn’t just need an answer, but how confident the model is.

    Statistical grounding → You can justify algorithms as maximum likelihood or Bayesian inference.

    Flexibility → Probabilistic models allow us to update beliefs when new data arrives.

# Probabilistic Interpretations of Common Models

This document gives brief probabilistic interpretations for three common models: Linear Regression (with Gaussian noise), Logistic Regression, and Naïve Bayes. Each model is shown with its key equation and a short explanation.

---

## 1. Linear Regression (with Gaussian noise)

**Equation**

For each training example $i$:

$$
y^{(i)} = \theta^\top x^{(i)} + \varepsilon^{(i)}, \qquad
\varepsilon^{(i)} \sim \mathcal{N}(0, \sigma^2).
$$

**Probabilistic interpretation**

For a given input $x^{(i)}$, the model asserts that the target $y^{(i)}$ is normally distributed with mean $\theta^\top x^{(i)}$ and variance $\sigma^2$. In density form:

$$
p(y \mid x; \theta) = \mathcal{N}\big(y \mid \theta^\top x,\, \sigma^2 \big).
$$

Under the IID Gaussian noise assumption, the likelihood of the dataset $\{(x^{(i)},y^{(i)})\}_{i=1}^m$ is

$$
L(\theta) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi}\,\sigma}
\exp\!\left(-\frac{(y^{(i)} - \theta^\top x^{(i)})^2}{2\sigma^2}\right).
$$

Maximizing this likelihood (or equivalently the log-likelihood) is equivalent to minimizing the sum-of-squares (least-squares) cost.

---

## 2. Logistic Regression

**Model / prediction**

Logistic regression models the probability of the binary label $y\in\{0,1\}$ as:

$$
P(y=1 \mid x; \theta) = \sigma(\theta^\top x) = \frac{1}{1 + e^{-\theta^\top x}}.
$$

**Probabilistic interpretation**

The conditional distribution of $y$ given $x$ is Bernoulli with parameter $\sigma(\theta^\top x)$:

$$
p(y \mid x; \theta) = \big(\sigma(\theta^\top x)\big)^y \big(1-\sigma(\theta^\top x)\big)^{1-y}.
$$

Thus logistic regression outputs a probability for the positive class (e.g., "spam with probability 0.92"), and training typically proceeds by maximizing the Bernoulli log-likelihood (or equivalently minimizing cross-entropy).

---

## 3. Naïve Bayes

**Model / formula**

Naïve Bayes applies Bayes' theorem with a conditional independence assumption across features:

$$
P(y \mid x) \propto P(y)\prod_{j} P(x_j \mid y).
$$

**Probabilistic interpretation**

Naïve Bayes is fully probabilistic: it models the joint distribution $p(x,y)$ by combining a prior $P(y)$ with class-conditional likelihoods $P(x_j \mid y)$. The “naïve” assumption is that features $x_j$ are conditionally independent given the class $y$. Classification chooses the class $y$ that maximizes the posterior $P(y\mid x)$.

---

## Notes & connections

- Many common loss functions arise from assuming a particular likelihood:
  - Gaussian noise $\rightarrow$ least-squares (L2).
  - Bernoulli likelihood $\rightarrow$ logistic / cross-entropy loss.
  - Laplace noise $\rightarrow$ L1 (absolute error).
- The **probabilistic interpretation** gives us uncertainty estimates, principled loss functions (via MLE), and a natural way to incorporate priors (Bayesian methods).

---
**General Pattern**

Many common ML algorithms can be derived as maximum likelihood estimators under certain probabilistic assumptions:

Linear regression → assumes Gaussian noise

Logistic regression → assumes Bernoulli likelihood

Poisson regression → assumes Poisson likelihood

So:

The “probabilistic interpretation” is the viewpoint that our ML model is describing a distribution over the data, and the algorithm we use (like least squares, logistic loss, etc.) naturally falls out from maximizing the probability (likelihood) of the observed data.

---

# Likelihood: A Function of Parameters Given Fixed Data

## What It Means

Think of two roles:

- **Data $D$**: what you already observed (fixed).
- **Parameters $\theta$**: knobs of your model you can vary.

The **likelihood** is the mapping

$$
L(\theta) = p(D \mid \theta),
$$

i.e., *for each candidate $\theta$, how plausible is the fixed data $D$ under the model?*  

So the variable is $\theta$; the data are held constant.

---

## Step-by-Step Explanation

**Step 0 — Freeze the data**  
You’ve collected $m$ examples $D = \{(x^{(i)}, y^{(i)})\}_{i=1}^m$. Treat these as constants.

**Step 1 — Choose a model family**  
Example: linear regression with Gaussian noise:

$$
y^{(i)} \mid x^{(i)}, \theta \sim \mathcal{N}(\theta^\top x^{(i)}, \sigma^2)
$$

**Step 2 — Score the data for a single $\theta$**  

$$
L(\theta) = \prod_{i=1}^m p(y^{(i)} \mid x^{(i)}; \theta)
$$

(This product comes from the IID assumption.)

**Step 3 — Vary $\theta$**  
Change $\theta$, recompute $L(\theta)$. Doing this for many $\theta$ values gives you a *function* from parameters to likelihood: $\theta \mapsto L(\theta)$.

**Step 4 — Pick the best $\theta$**  
Choose the $\theta$ that **maximizes** $L(\theta)$ (or equivalently, the log-likelihood $\ell(\theta) = \log L(\theta)$). This is **Maximum Likelihood Estimation (MLE)**.

> Key distinction: Probability treats $\theta$ as fixed and asks about data;  
> Likelihood treats **data as fixed** and asks about $\theta$.

---

## Tiny Concrete Example: Coin Toss

Data: 5 tosses → HHTHH (4 heads, 1 tail)  
Model: Bernoulli($\theta$) per toss, IID.

$$
L(\theta) = \theta^k (1 - \theta)^{n-k} = \theta^4 (1 - \theta)^1
$$

Evaluate at a few $\theta$ values:

| θ     | L(θ)         |
|-------|--------------|
| 0.2   | 0.00128      |
| 0.5   | 0.015625     |
| 0.8   | 0.08192 ← largest |

The **MLE** is $\hat{\theta} = k/n = 0.8$.

---

## Linear Regression Tie-In

With Gaussian noise:

$$
p(y^{(i)} \mid x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\Big(-\frac{(y^{(i)} - \theta^\top x^{(i)})^2}{2\sigma^2}\Big)
$$

Thus, log-likelihood:

$$
\ell(\theta) = \log L(\theta) = \text{const} - \frac{1}{2\sigma^2} \sum_{i=1}^m (y^{(i)} - \theta^\top x^{(i)})^2
$$

Maximizing $\ell(\theta)$ is **equivalent** to minimizing the **sum of squared errors**.  
This is why **least squares** naturally arises from a probabilistic viewpoint.

---

## Practical Notes

- **Likelihood isn’t a probability over $\theta$**. It needn’t integrate to 1; it just *ranks* parameter values by how well they explain the fixed data.
- We often optimize the **negative log-likelihood (NLL)** because sums are easier than products, and logs eliminate constants that don’t depend on $\theta$.

---

# Probabilistic View of Linear Regression

## Setup

We assume a **linear model with noise**:

$$
y^{(i)} = \theta^T x^{(i)} + \varepsilon^{(i)}
$$

where:

- $\theta$ = model parameters (weights)  
- $x^{(i)}$ = input features  
- $y^{(i)}$ = observed target  
- $\varepsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$ = Gaussian noise  

This means: the true output is a linear function of the inputs **plus Gaussian noise**.

---

## Distribution of $y^{(i)}$

Because $\varepsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$, we get:

$$
y^{(i)} \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2).
$$

So, the **conditional distribution** of $y^{(i)}$ given $x^{(i)}$ is Gaussian with:

- Mean = $\theta^T x^{(i)}$ (our linear prediction)  
- Variance = $\sigma^2$ (spread due to noise)  

---

## Likelihood of Data

If we have $m$ training examples, the **likelihood** of observing all the $y^{(i)}$ given inputs and parameters is:

$$
L(\theta) = \prod_{i=1}^{m} p(y^{(i)} \mid x^{(i)}; \theta).
$$

Since each $y^{(i)}$ is Gaussian:

$$
p(y^{(i)} \mid x^{(i)}; \theta) =
\frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right).
$$

Thus:

$$
L(\theta) =
\prod_{i=1}^{m} \frac{1}{\sqrt{2\pi}\sigma}
\exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right).
$$

---

## Log-Likelihood

It’s easier to work with the log-likelihood:

$$
\ell(\theta) = \log L(\theta)
= -\frac{m}{2}\log(2\pi\sigma^2)
- \frac{1}{2\sigma^2}\sum_{i=1}^{m}(y^{(i)} - \theta^T x^{(i)})^2.
$$

---

## Maximum Likelihood Estimation (MLE)

Maximizing $\ell(\theta)$ w.r.t. $\theta$ is equivalent to minimizing:

$$
J(\theta) = \sum_{i=1}^{m}(y^{(i)} - \theta^T x^{(i)})^2.
$$

That’s exactly the **least-squares cost function**.

---

## Interpretation

- Least-squares regression is not just a random choice.  
- It is the **Maximum Likelihood Estimator (MLE)** under the assumption:  
  1. The errors are **i.i.d. Gaussian**  
  2. Mean zero, constant variance ($\sigma^2$).  

So **linear regression with least-squares loss** comes naturally from a **probabilistic model of the data**.
