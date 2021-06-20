# Monotonic Neual Networks In Keras
A simple(r) way to obtain guaranteed monotonicity and/or convexity in Keras.

by Ben Guiles

Neural networks, especially deep ones, have a reputation for being opaque to human understanding and for delivering predictions that are difficult to explain. They are, after all, large multivariate functions consisting of often thousands of compositions of other multivariate functions. While it’s easy to explain what each piece of the neural network does alone, it is not trivial to wrap your head around why or how a neural network delivers a particular output given a set of inputs.

The field of AI receives a lot of criticism for this. Much of it is deserved. AI engineers and data scientists have failed to sufficiently explain why such intractable mathematical equations should be trusted in critical situations involving life, death, livelihood, college admissions, loan applications, criminal justice, etc. 

A subset of the problems for which neural networks have difficulty providing satisfactory results are those where an AI must deliver an answer that is not only right most of the time, but reasonable ALL of the time. For this subset, it is possible to apply hard constraints to a model that will guarantee that a certain relationship between input and output variables is maintained.

For instance, in college admissions, property value assessments, loan applications, and other problems it is reasonable to expect certain features should have a monotonic--i.e. always increasing or always decreasing--relationship with the output variable. 

A neural network trained to predict graduation probability from GPA and SAT scores may, without the application of a monotonicity constraint, mistakenly predict lower graduation probabilities for higher GPAs or SAT scores over certain combinations of inputs.

Luckily, the mathematical structure of a neural network does allow us to apply a hard monotonicity constraint. Remember that a neural network is itself a function that is the composition of many smaller functions. When we compose monotonic functions together, we get another monotonic function! Applied in the right way, we can design a neural network whose output is monotonic with respect to any or all of the inputs, while maintaining all of the flexibility of an unconstrained neural network. We can even enable the network to learn whether or not the monotonicity should be increasing or decreasing based on the data alone.

# Monotonic Activation Function

This method relies on using either a single, nonconvex monotonic activation function, or a combination of two activation functions which are both monotonic and convex upward (i.e. leaky ReLU) and monotonic and convex downward. Neither of these are provided completely in Keras, but Keras does make it easy to define our own.

### Wiggle

Wiggle is the identity asymptote: wiggle(x) approaches x as |x| approaches infinity. Is is nonconvex in both the positive and negative domains. 

You can see it in action here: https://www.desmos.com/calculator/4jrq6iwfzg


```
def wiggle(x):
    return x**3 / (x**2 + 1)
```


### Leaky ReLU and Leaky NeLU

Leaky ReLU is defined by relu(x) = max(kx,x), where k is a constant less than one, usually 0.1. It is monotonically increasing and convex upward.

Leaky NeLU is defined by nelu(x) = min(kx,x). It is monotonically increasing and convex downward.

Leaky ReLU and Leaky NeLU are both monotonic, but they are also convex. If we compose a convex function with another function that is convex in the same direction, the result can only be another convex function. If we want to constrain our neural network to be monotonic, but not necessarily convex, we need to use both activations functions that are convex upward and convex downward. As long as both are monotonic, the output of the combination and composition of both will always be monotonic as well.


```
def nelu(x, alpha=0.1, max_value=None, threshold=0):
  return -1 * K.relu((-1 * x), alpha=alpha, max_value=max_value, threshold=threshold)
```

The advantage of wiggle(x) is that it is a single function, and much simpler to use. However, it is more difficult to differentiate and calculate than relu(x)/nelu(x), so some minor training and inference performance gains may be realized with the latter. In my experience, however, training and inference time is more impacted by the number of memory operations required than the floating point operations.

A second advantage of using a combination of relu(x) and nelu(x) is that the "tails" of the function approximated by the neural network can go in independent directions, potentially improving the ability of the model to extrapolate reasonable, if not perfectly accurate answers outside of the original domain of the training data.
