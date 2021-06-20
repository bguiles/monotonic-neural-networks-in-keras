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

Wiggle is the identity asymptote: wiggle(x) approaches x as |x| approaches infinity. Is is nonconvex in both the positive and negative domains. You can see it in action here:



```
def wiggle(x):
    return x**3 / (x**2 + 1)
```
