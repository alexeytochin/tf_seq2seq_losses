"""Finite difference Jacobian approximation for batched functions."""

# ==============================================================================
# Copyright 2021 Alexey Tochin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Callable, Union
import tensorflow as tf


def finite_difference_batch_jacobian(
    func: Callable[[tf.Tensor], tf.Tensor], x: tf.Tensor, epsilon: float
) -> tf.Tensor:
    """Calculate final difference Jacobian approximation
        gradient : x_bij -> gradient(x)_bij = (func(x + epsilon)_bij - func(x)_bij) / epsilon
    for given function
        func : x_bij -> y_bkl = func(x)_bkl.

    The first (zero) dimension of x is the batch dimension. Namely, it is assumed that different components of func(x)_b
    are independent functions of different components of x_b. For example:
        func = tf.reduce_sum(x**2, axis = [1,2]), for rank = 3 tensor x,
    but not
        incorrect_func = tf.reduce_sum(x**2, axis = [0, 1, 2]).

    Example of usage:
    ```python
        x = tf.ones(shape=[2, 3, 4])
        epsilon = 1e-3
        func = lambda x: tf.reduce_sum(x**2, axis=[1,2]) / 2
        finite_difference_gradient(func=func, x=x, epsilon=epsilon)
        # -> tf.ones(shape=[2, 3, 4], dtype=tf.float32) # (approximately for epsilon -> 0)
    ```

    Args:
        func:       Callable:   tf.Tensor, shape = [batch_size] + DIMS_y,   dtype = tf.float32
                            ->  tf.Tensor, shape = [batch_size] + DIMS_x,   dtype = tf.float32
        x:          tf.Tensor, shape = [batch_size] + DIMS_x,               dtype = tf.float32
                    that is the input tensor for func
        epsilon:    float, finite difference parameter.

    Returns:        tf.Tensor, shape = [batch_size] + DIMS_x + DIMS_y
    """
    y0 = func(x)
    dims_y = tf.shape(y0)[1:]
    dims_x = tf.shape(x)[1:]
    batch_size = tf.shape(x)[0]
    x_reshaped = tf.reshape(x, [batch_size, -1])
    # shape: [batch_size, dim_x]

    def func_(x_: tf.Tensor) -> tf.Tensor:
        """Argument for _finite_difference_batch_jacobian

        Args:
            x_:     shape: [batch_size, dim_x]

        Returns:    shape: [batch_size, dim_y]
        """
        x_orig = tf.reshape(x_, shape=tf.concat([[-1], dims_x], axis=0))
        # shape = [batch_size] + DIM_x
        y_orig = func(x_orig)
        # shape = [batch_size] + DIM_y
        y_reshaped = tf.reshape(y_orig, [tf.shape(x_)[0], -1])
        # shape = [batch_size, dim_y]
        return y_reshaped

    dy_reshaped = _finite_difference_batch_jacobian(
        func=func_, x=x_reshaped, epsilon=epsilon
    )
    # shape: [batch_size] + DIMS_y + DIMS_x
    dy_shape = tf.concat([tf.expand_dims(batch_size, 0), dims_y, dims_x], axis=0)
    dy = tf.reshape(dy_reshaped, shape=dy_shape)
    # shape: [batch_size] + DIMS_y + DIMS_x

    return dy


def _finite_difference_batch_jacobian(
    func, x, epsilon: Union[float, tf.Tensor]
) -> tf.Tensor:
    """

    Args:
        func:   shape = [batch_size, dim_x] -> [batch_size, dim_y]
        x:      shape = [batch_size, dim_x]

    Returns:    shape = [batch_size, dim_y, dim_x]
    """
    dim_x = tf.shape(x)[1]
    dx = tf.expand_dims(tf.eye(dim_x, dtype=x.dtype), axis=1) * epsilon
    # shape: [dim_x, 1, dim_x]
    pre_x1 = tf.expand_dims(x, 0) + dx
    # shape: [dim_x, batch_size, dim_x]
    y0 = func(x)
    # shape: [batch_size, dim_y]
    dy_transposed = (
        tf.vectorized_map(fn=func, elems=pre_x1) - tf.expand_dims(y0, 0)
    ) / epsilon
    # shape: [dim_x, batch_size, dim_y]
    dy = tf.transpose(dy_transposed, perm=[1, 2, 0])

    return dy
