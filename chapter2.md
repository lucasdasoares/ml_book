# Supervised Learning

Recall the house-price predictor. How does a model actually *learn* to turn square footage and number of bedrooms into a price? As we saw, supervised learning models define a mapping from input data to output predictions based on provided examples of input and output pairs. As said before, supervised learning models define a mapping from input data to output predictions based on provided examples of input and output pairs.

Mathematically, we can represent a supervised learning model as a function $f(\mathbf{x})$ that takes inputs $\mathbf{x}$, described in the form of a vector, and returns an output vector $\mathbf{\hat{y}}$:

$$
\mathbf{\hat{y}} = f(\mathbf{x})
$$

Thus, the model is just a mathematical equation with a fixed form. It represents a family of different relations between the inputs and outputs. The model also contains **parameters** $\boldsymbol{\theta}$, such that the choice of parameters determines the particular relation between inputs and outputs, so we can write

$$
\mathbf{\hat{y}} = f(\mathbf{x}, \boldsymbol{\theta})
$$

When we talk about learning or training a model, we mean that we attempt to find the set of parameters $\boldsymbol{\theta}$ that make good output predictions from the input, that is, we want to find the equation that describes the training data most accurately. For example, for a linear model with two independent variables, we have

$$
\hat{y} = w_1 x_1 + w_2 x_2 + b,
$$

where $x_1$ and $x_2$ are the independent variables and $\mathbf{x} = \left[ x_1, x_2 \right]^T$ is the vector of independent variables, or input vector, and $w_1$, $w_2$, and $b$ are the model parameters, such that $\boldsymbol{\theta} = \left[ w_1, w_2, b \right]^T$ is the vector of parameters.

We learn these parameters using a **training dataset** of pairs of input and output examples $\left\{ \mathbf{x}_n, \mathbf{y}_n \right\}_{n=1}^{N}$. Our objective is to select the parameters $\boldsymbol{\theta}$ that map each training input to its associated output as closely as possible.

We quantify the degree of mismatch in this mapping with a **loss function** $J$, also known as **cost function** or **objective function**. This is a scalar value that summarizes how poorly the model predicts the training outputs from their corresponding input for the model parameters.

For instance, in a regression problem a very common choice is the **mean squared error**, which averages the squared gap between each prediction and its true output:

$$
J(\boldsymbol{\theta}) = \frac{1}{N} \sum_{n=1}^{N} \left( y_n - f(\mathbf{x}_n, \boldsymbol{\theta}) \right)^2.
$$

The squaring makes every error positive and penalizes large mistakes more heavily than small ones. Minimizing $J$ therefore pushes the predictions $f(\mathbf{x}_n, \boldsymbol{\theta})$ toward the true values $y_n$.

We can then consider the loss as a function of the parameters $J(\boldsymbol{\theta})$, such that, when we train the model, we are seeking the parameters that minimize this loss function:

$$
\hat{\boldsymbol{\theta}} = \operatorname*{argmin}_{\boldsymbol{\theta}} J(\boldsymbol{\theta})
$$

If the loss is small after this minimization, we have found model parameters that accurately predict the training outputs $\mathbf{y}_n$ from the training inputs $\mathbf{x}_n$.

After training a model, we assess it on data it did not observe during training. We use a **validation set** to compare candidate models and settings and to check for problems such as overfitting, and finally a separate **test set** to obtain an unbiased estimate of how well the chosen model generalizes. If this estimate is adequate, we are ready to deploy the model.

## Model Evaluation

In supervised learning, in both regression and classification problems, each model represents a family of equations mapping inputs to outputs. The particular equation is chosen using the **training dataset**, which are examples of input/output pairs used to choose the best model parameters.

The process of training a supervised learning model involves using a learning algorithm to adjust the model's parameters so that its predictions closely match the true outcomes in the training data.

During development, we evaluate candidate models on a separate **validation dataset** that the model did not see during training. This lets us compare models, tune their settings, and detect underfitting or overfitting. Because these choices are guided by the validation set, its score is no longer an unbiased estimate of real-world performance. For that final estimate we hold out a third **test dataset**, which we touch only once, after all choices have been fixed.

With this in mind, we split our data into three parts in all applications throughout this course: the **training dataset** for adjusting the model parameters, the **validation dataset** for comparing models and choosing their settings, and the **test dataset** for the final, unbiased evaluation of the chosen model's performance.

## Underfitting and Overfitting

Having trained the model, we want to know whether it has captured the real structure of the data or merely memorized it. We diagnose this by comparing its loss on the training set with its loss on the validation set. The degree to which the prediction accuracy generalizes to the validation data depends in part on how representative and complete the training data is. However, it also depends on how expressive the model is.

A simple model like a line might not be able to capture the true relationship between input and output. This is known as **underfitting**. It is straightforward to identify an underfit model: it will have a poor performance both on the training dataset and on the validation dataset.

Conversely, a very expressive model may describe statistical peculiarities of the training data that are atypical and lead to unusual predictions. This is known as **overfitting**. In this case, the model overlearns a dataset, to the extent of learning all the noise and errors in a dataset. Thus, the model fails to perform well for unobserved data.

We can detect if a model is overfitting the training data by comparing its performance on both training dataset and validation dataset. If the model has a good performance on the training dataset, but a poor performance on the validation dataset, then it is probably overfitting to the training data.

![Underfitting, appropriate fit, and overfitting](../images/underfitting_overfitting_01.png)

***Left:** the model is too simple to capture the structure of the data (underfitting). **Center:** an appropriate fit. **Right:** the model has fit the noise in the data (overfitting).*

In the figure on the left, the simplicity of the model did not allow it to capture the data structure, indicating a case of underfitting. In the center figure, we have an ideal fit. In the figure on the right, we observe that the model started to fit to the noise present in the data, representing a case of overfitting.

<a id="fig-complexity-curve"></a>

![Training and validation loss versus model complexity](../images/complexity_curve.png)

*Figure: As model complexity increases, training loss keeps falling while validation loss falls and then rises. The gap between the two curves widens in the overfitting regime.*

This figure summarizes the tension between underfitting and overfitting in a single picture. The horizontal axis represents **model complexity** — roughly, how expressive the model is, or how many parameters it can use to bend itself to the data. The vertical axis shows the **loss**, so lower is better.

As complexity increases, the **training loss** (solid curve) falls steadily: a more expressive model can always fit the training examples at least as well as a simpler one, and usually better. If we judged our model by training loss alone, we would be tempted to keep making it more and more complex.

The **validation loss** (dashed curve) tells a different story. On the left, where the model is too simple, it cannot capture the underlying structure of the data, so it predicts poorly on *both* the training and the validation sets. This is the **underfitting** regime, and both curves are high. As complexity grows, the model starts to capture the true pattern and the validation loss falls, tracking the training loss closely.

At some point the validation loss reaches a minimum. This is the **good fit** — the sweet spot where the model is expressive enough to capture the real structure of the data but not so expressive that it starts memorizing its peculiarities.

Beyond that point the two curves diverge. The training loss keeps dropping, but the validation loss begins to *rise*: the extra capacity is now being spent fitting the noise and idiosyncrasies of the training set, which do not carry over to unseen data. This is the **overfitting** regime. The growing vertical gap between the two curves is a direct measure of how much the model has overfit — a large gap means the model performs far better on data it has seen than on data it has not.

The practical lesson is that we cannot choose model complexity by minimizing the training loss; if we could, we would always pick the most complex model available. Instead, we watch the validation loss and aim for the bottom of its U-shaped curve, where the model generalizes best to new data.
