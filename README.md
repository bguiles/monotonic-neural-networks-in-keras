# Building Monotonic Deep Neural Networks Using Keras

###### A simple(r) way to build monotonic neural networks in Keras.

###### by Ben Guiles

Neural networks have earned a bit of a reputation for being black boxes. Opaque and incomprehensible to human understanding, they detect inscrutable patterns in huge data sets and then, as if by magic, return lots of very good predictions, among a few utterly head-scratching ones. The endurance of this perception is understandable. Neural networks are, after all, very large multivariate functions consisting of often thousands of compositions of other multivariate functions. While it’s easy to explain what each piece of the neural network does alone, it is not at all trivial to wrap your head around why or how a neural network delivers a particular output given a set of inputs.

The intractability of neural networks underlying most AI systems is a consistent source of criticism of the field. This isn't entirely unfair. AI engineers and data scientists have failed to sufficiently demonstrate that incomprehensible mathematical equations should be trusted over human judgment in critical situations involving life, death, livelihood, college admissions, loan applications, criminal justice, etc. 

That isn't because those networks haven't achieved state of the art metrics according to the industry standard loss functions. Rather, the problems that those networks aim to solve require answers that are not only right most of the time, but are at a minimum, consistent and reasonable ALL of the time. Unfortunately, "reasonableness" is a tricky metric to pin down. For a few of these problems however, we can frame the reasonableness of a result as following a common sense heuristic, which we might be able translate into properties of mathematical functions. 

For example:

"If x goes up, y should always go up."

Translation: f(x) shall be monotonically increasing

"As x approaches zero, y should approach infinity, and as x approaches infinity, y should approach zero." 

Translation: exp(f(log(x))) shall be monotonically decreasing

"As input 2 goes up, output should go down, and the rate at which output goes down should only decrease." 

Translation: f(x) shall be monotonically decreasing and convex upwards

In each of these cases, if we can guarantee monotonicity, we can achieve a reasonable result. Fortunately, it is possible to guarantee a monotonic output from a neural network with respect to any or all of its inputs, while maintaining the unconstrained flexibility which makes neural networks so valuable for modeling extremely complex problems.

Remember that a neural network is itself a function that is the composition of many smaller functions. When we compose monotonic functions together, we get another monotonic function! Applying this principle in the right way, we can design a neural network whose output is monotonic with respect to any or all of the inputs, while maintaining all of the flexibility of an unconstrained neural network. We can even enable the network to learn whether or not the monotonicity should be increasing or decreasing based on the data alone.

We can even do this without any real difficulty using Keras, one of the most popular tools for building and training deep neural networks in the deep learning framework Tensorflow.

The concept is simple: monotonic activation functions + careful placement of sign constraints to layer weights = guaranteed monotonicity

# Monotonic Activation Functions

This method relies on using either a single, nonconvex monotonic activation function, or a combination of two activation functions in parallel which are both monotonically increasing & convex upward (i.e. leaky ReLU) and monotonically increasing & convex downward. Neither of these options are provided completely in Keras, but Keras does make it easy to roll our own activation functions.

### Wiggle

Wiggle(x) approaches x as x approaches +/- infinity, but it has a little, you know, wiggle where x=0, making it nonlinear. Is is nonconvex in both the positive and negative domains.

You can see it in action here: https://www.desmos.com/calculator/4jrq6iwfzg

```
def wiggle(x):
    return x**3 / (x**2 + 1)
```

### Leaky ReLU and Leaky NeLU

Leaky ReLU is defined by relu(x) = max(kx,x), where k is a constant less than one, usually 0.1. It is monotonically increasing and convex upward.

Leaky NeLU is defined by nelu(x) = min(kx,x). It is monotonically increasing and convex downward. NeLU doesn't stand for anything in particular, but it implies inversion.

Leaky ReLU and Leaky NeLU are both monotonic, but they are also convex. If we compose a convex function with another function that is convex in the same direction, the result can only be another convex function. If we want to constrain our neural network to be monotonic, but not necessarily convex, we need to use both activations functions that are convex upward and convex downward. As long as both are monotonic in the same direction, the output of the combination and composition of both will always be monotonic as well.

Here you can see that nelu(x) is just a linear transformation of (leaky) relu(x):

```
# Keras comes packaged with Leaky ReLU already implemented. We can tweak it a little bit to get nelu(x).

from keras import backend as K

def nelu(x, alpha=0.1, max_value=None, threshold=0):
  return -1 * K.relu((-1 * x), alpha=alpha, max_value=max_value, threshold=threshold)
```

The advantage of wiggle(x) is that it is a single function, and much simpler to use with the Keras sequential and functional APIs. To use relu(x)/nelu(x) as parallel activation functions, we need to use tf.split operations and separate activation layers in the functional API. However, wiggle(x) requires more computing power to differentiate and calculate than relu(x)/nelu(x), so some minor training and inference performance gains may be realized with the latter. In my experience, however, training and inference time is more impacted by the number of memory operations required than the floating point operations, so the difference is minimal for smaller models.

A second advantage of using a combination of relu(x) and nelu(x) is that the "tails" of the function approximated by the neural network can go in independent directions, potentially improving the ability of the model to extrapolate reasonable, if not perfectly accurate inferences outside of the original domain of the training data.

In practice, either approach can achieve a monotonic neural network.

# Same Sign Weight Constraints

Using monotonic activation functions is not enough to ensure monotonicity between an input and an output of a neural network. We will also need to constrain the sign of the weights of each layer.

To do this, we create a SameSign class, subclassed from tf.keras.constrains.Constraint. This class is applied to a Keras layer as either the kernel\_constraint or bias\_constraint argument. While there is no strict reason we can't use this with bias\_constraint, biases to not need to be constrained to all positive or all negative to ensure monotonicity. So we'll ignore those.

The Constraint class is called by a layer every training epoch _after_ the weights are updated by the optimizer, but _before_ those weights are used to make predictions on the next batch. A Constraint can be applied to a layer using the kernel\_constraint argument to constrain weights or bias\_constraint argument to constrain biases. 

(While there is no strict reason we can't use SameSign with the bias\_constraint, biases to not need to be constrained to all positive or all negative to ensure monotonicity. So we'll ignore those for the time being.)

The SameSign class works like this: Each layer is initialized with an axis and array of inputs which will be forced to share a sign. For the first hidden layer of a neural network, set the axis to 1 to 

```
class SameSign(tf.keras.constraints.Constraint):

  def __init__(self, axis=None, mono_inputs=1.):
    self.axis = axis               
    self.mono_inputs = tf.cast(mono_inputs, dtype=tf.float32)  # Apply SameSign constraint only to the inputs that should be monotonic w/r/t output. mono_inputs is a boolean array (e.g. [[0],[1]]) the same length as the input vector.

  def __call__(self, w):
    w_mean = tf.math.reduce_mean(w, axis=self.axis, keepdims=True)
    m_pos = tf.cast(tf.math.greater_equal(w_mean,0.0), dtype=tf.float32)
    m_neg = tf.cast(tf.math.less(w_mean,0.0), dtype=tf.float32)

    # Conditional form of the SameSign algorithm
    '''
    if w_mean >= 0:
      return abs(w)
    else:
      return -1 * abs(w)
    '''

    # SameSign algorithm without conditionals
    return (tf.cast(tf.math.abs(self.mono_inputs-1), dtype=tf.float32) * w) + (tf.cast(self.mono_inputs, dtype=tf.float32) * ((m_pos * abs(w)) + (m_neg * -1. * abs(w))))
```
