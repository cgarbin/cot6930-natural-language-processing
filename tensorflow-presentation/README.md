# TensorFlow introduction for natural language processing

This is a simple introduction to TensorFlow, with emphasis in natural language
processing (NLP) applications.

It is divided in two parts:

1. What TensorFlow is
1. Applying TensorFlow to natural language process (NLP)

This introduction has a companion slide deck [in this file](<./COT-6930 presentation - TensorFlow.pdf>).

## TensorFlow introduction

### What is a tensor?

_Tensor_ is a n-dimensional data structure, where `n` can be any number greater
than or equal to zero. The number of dimensions is the _rank_ of the tensor.

Examples of tensors:

| Rank 0                                       | Rank 1                                       | Rank 2                                       | Rank n                                       |
| -------------------------------------------- | -------------------------------------------- | -------------------------------------------- | -------------------------------------------- |
| 0D tensor                                    | 1D tensor                                    | 2D tensor                                    | nD Tensor                                    |
| Scalar                                       | Vector                                       | Matrix                                       | Tensor                                       |
| ![Rank 0](./images/tensor-example-rank0.png) | ![Rank 1](./images/tensor-example-rank1.png) | ![Rank 2](./images/tensor-example-rank2.png) | ![Rank n](./images/tensor-example-rankn.png) |

### What is TensorFlow?

TensorFlow is a generic, high-performance computation engine.

It represents computations as graphs.

For example, this computation:

    a = 2
    b = 2
    c = 1

    d = b + c
    e = a + c

    result = d * e

Is represented in TensorFlow as:

    a = tf.constant(2.0, name='a')
    b = tf.constant(2.0, name='b')
    c = tf.constant(1.0, name='c')

    d = tf.add(b, c, name='b_plus_c')
    e = tf.add(a, c, name='a_plus_c')

    result = tf.multiply(d, e, name='d_times_e')

Which results in this graph:

![Computation Graph](./images/what-is-tf-simple-graph.png)

The nodes are the computations to be performed and the edges is where data (the
tensors) flows from one edge to another (and now we have the pieces for the
name: _TensorFlow_).

Putting it all together: the computation, the TensorFlow equivalent and its
graph showing the nodes (computations) and edges (where data flows from one
node to the next):

![Computation Graph](./images/what-is-tf-all-together.png)
