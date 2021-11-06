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

from typing import Union, Callable
import tensorflow as tf
import numpy as np


def logit_to_logproba(logit: tf.Tensor, axis: int) -> tf.Tensor:
    """Converts logits to logarithmic probabilities:
        logit_to_logproba(x) = x - log (sum along axis (exp(x))

    Args:
        logit:  tf.Tensor, dtype = tf.float32
        axis: integer, like for tf.reduce_logsumexp

    Returns:    tf.Tensor, of the same shape and size as input logit
    """
    log_probas = logit - tf.reduce_logsumexp(input_tensor=logit, axis=axis, keepdims=True)
    return log_probas


def apply_logarithmic_mask(tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Masks a logarithmic representation of a tensor, namely
    1. Keeps the value of tensor unchanged for True values of mask
    2. Replace the value of tensor by -tf.inf for False values of mask

    Args:
        tensor: tf.Tensor, dtype = tf.float32 of the same shape as mask or broadcastable
        mask:   tf.Tensor, dbool = tf.float32 of the same shape as tensor or broadcastable

    Returns:    tf.Tensor, dtype = tf.float32 of the same shape as tensor
    """
    return tensor + tf.math.log(tf.cast(mask, dtype=tf.float32))


def pad_with_minus_inf(tensor: tf.Tensor, desired_size: int, axis: int) -> tf.Tensor:
    return pad_until(
        tensor=tensor,
        desired_size=desired_size,
        axis=axis,
        pad_value=tf.constant(-np.inf, dtype=tensor.dtype),
    )


def logsumexp(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """A numerically stable version of elementwise function
        logsumexp(x, y) = log (e ** x + e ** y)

    Args:
        x:      tf.Tensor of the shape and size as y or broadcastable
        y:      tf.Tensor of the shape and size as x or broadcastable

    Returns:    tf.Tensor of the shape and size as x and y
    """
    return tf.where(
        condition=x < y,
        x=y + tf.math.softplus(x - y),
        y=tf.where(
            condition=x > y,
            x=x + tf.math.softplus(y - x),
            y=x + np.log(2.)
        ),
    )


def subexp(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """A numerically stable version of elementwise function
        subexp(x,y) := exp x - exp y

    Args:
        x:      tf.Tensor, shape broadcastable to y
        y:      tf.Tensor, shape broadcastable to x

    Returns:    tf.Tensor, shape, the same as x and y
    """
    return tf.where(
        condition=x > y,
        x=-tf.exp(x) * tf.math.expm1(y - x),
        y=tf.where(
            condition=x < y,
            x=tf.exp(y) * tf.math.expm1(x - y),
            y=tf.zeros_like(x),
        ),
    )


def unsorted_segment_logsumexp(data: tf.Tensor, segment_ids: tf.Tensor, num_segments: Union[int, tf.Tensor])\
        -> tf.Tensor:
    """Computes the logarithmic sum of exponents along segments of a tensor
    like other operators from tf.math.unsorted_segment_* family.

    Args:
        data:           tf.Tensor,  shape = [...] + data_dims,
        segment_ids:    tf.Tensor,  shape = [...], dtype = tf.int32
        num_segments:   tf.Tensor,  shape = [], dtype = tf.int32

    Returns:            tf.Tensor,  shape = [num_segments] + data_dims, for the same type as data
    """
    data_max = tf.math.unsorted_segment_max(data=data, segment_ids=segment_ids, num_segments=num_segments)
    data_normed = data - tf.gather(params=data_max, indices=segment_ids)
    output = data_max + tf.math.log(tf.math.unsorted_segment_sum(
        data=tf.exp(data_normed),
        segment_ids=segment_ids,
        num_segments=num_segments,
    ))
    return output


def pad_until(
        tensor: tf.Tensor,
        desired_size: Union[tf.Tensor, int],
        axis: int,
        pad_value: Union[tf.Tensor, int, float, bool] = 0
) -> tf.Tensor:
    """Pads tensor until desired dimension from right,

    Args:
        tensor:         tf.Tensor, of any shape and type
        desired_size:   tf.Tensor or pythonic static integer
        axis:           pythonic static integer for pad axes
        pad_value:      tf.Tensor or pythonic numerical for padding

    Returns:            tf.Tensor, the same shape as tensor except axis that equals to desired_size.
    """
    rank = len(tensor.shape)
    if axis >= rank:
        raise ValueError()

    current_size = tf.shape(tensor)[axis]
    paddings = [[0, 0]] * axis + [[0, desired_size - current_size]] + [[0, 0]] * (rank - axis - 1)
    return tf.pad(tensor=tensor, paddings=paddings, constant_values=pad_value)


def insert_zeros(tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """
    Insert zeros into tensor before each masked element.
    For example:
    ```python
        output = insert_zeros(
            tensor =  tf.constant([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], dtype = tf.int32),
            mask = tf.constant([[False, True, False, False, True], [False, True,  True, True,  False]]),
        )
        # -> [[1, 0, 2, 3, 4, 0, 5, 0], [10, 0, 20, 0, 30, 0, 40, 50]]
        # We insert 0s 2, 5, 20, 30, and 40 because their positions in input tensor corresponds to True value
        in mask.
    ```

    Args:
        tensor: tf.Tensor, shape = [batch, length], any type and the same shape as mask
        mask:   tf.Tensor, shape = [batch, length], dtype = tf.bool and the same shape as tensor

    Returns:    tf.Tensor, shape = [batch, length + max_num_insertions],
                where max_num_insertions is the maximal number of True values along the 0 batch dimension of mask.
                dtype = same as input tensor
    """
    batch_size = tf.shape(tensor)[0]
    length = tf.shape(mask)[1]

    delta = tf.cumsum(tf.cast(mask, dtype=tf.int32), exclusive=False, axis=1)
    max_num_insertions = tf.reduce_max(delta[:, -1])

    y, x = tf.meshgrid(tf.range(length), tf.range(batch_size))
    y = y + delta
    indices = tf.reshape(tf.stack([x, y], 2), [-1, 2])

    output = tf.scatter_nd(
        indices=indices,
        updates=tf.reshape(tensor, shape=[-1]),
        shape=tf.stack([batch_size, length + max_num_insertions])
    )

    return output


def finite_difference_gradient(func: Callable[[tf.Tensor], tf.Tensor], x: tf.Tensor, epsilon: float) -> tf.Tensor:
    """Calculate final difference gradient approximation
        gradient : x_bij -> gradient(x)_bij = (func(x + epsilon)_bij - func(x)_bij) / epsilon
    for given function
        func : x_bij -> y_b = func(x)_b.

    The first (zero) dimension of x is the batch dimension. Namely it is assumed that different components of func(x)_b
    are independent functions of different components of x_b. For example:
        func = tf.reduce_sum(x**2, axis = [1,2]), for rank = 3 tensor x,
    but not
        incorrect_func = tf.reduce_sum(x**2, axis = [0,1,2]).

    Example of usage:
    ```python
        x = tf.ones(shape=[2, 3, 4])
        epsilon = 1e-3
        func = lambda x: tf.reduce_sum(x**2, axis=[1,2]) / 2
        finite_difference_gradient(func=func, x=x, epsilon=epsilon)
        # -> tf.ones(shape=[2, 3, 4], dtype=tf.float32) # (approximately for epsilon -> 0)
   ```

    Args:
        func:       Callable:   tf.Tensor, shape = [batch_size, ...],   dtype = tf.float32
                            ->  tf.Tensor, shape = [batch_size],        dtype = tf.float32
        x:          tf.Tensor, shape = [batch_size, ...], rank >= 1,    dtype = tf.float32
                    that is the input tensor for func
        epsilon:    float, finite difference parameter.

    Returns:        tf.Tensor, shape = [batch_size, ...] (the same shape as for x.)
    """
    input_shape = tf.shape(x)[1:]
    input_rank = input_shape.shape[0]
    dim = tf.reduce_prod(input_shape)
    dx = tf.reshape(
        tf.eye(dim, dtype=x.dtype),
        shape=tf.concat([tf.constant([1]), tf.reshape(dim, [1]), input_shape], axis=0)
    )
    # shape = [1, dim] + input_shape

    pre_x1 = tf.expand_dims(x, 1) + epsilon * dx
    # shape = [batch_size, dim] + input_shape
    x1 = tf.reshape(pre_x1, shape=tf.concat([tf.constant([-1], dtype=tf.int32), input_shape], axis=0))
    # shape = [batch_size * dim] + input_shape
    x0 = tf.tile(x, multiples=[dim] + [1] * input_rank)
    pre_derivative = (func(x1) - func(x0)) / epsilon
    # shape = [batch_size * dim]
    derivative = tf.reshape(pre_derivative, shape=tf.concat([tf.constant([-1]), input_shape], axis=0))
    # shape = [batch_size] + input_shape
    return derivative


def unfold(
        init_tensor: tf.Tensor,
        iterfunc: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        num_iters: Union[int, tf.Tensor],
        d_i: int,
        element_shape: tf.TensorShape,
        swap_memory: bool = False,
        name: str = "unfold",
) -> tf.Tensor:
    """Calculates a tensor by iterations over i that is the concatenation
        for d_i = +1:
            init_tensor
            iterfunc(init_tensor, 0)
            iterfunc(iterfunc(init_tensor, 0), 1)
            ...
            ..., num_iters - 1)
            ..., num_iters - 1), num_iters)
        for d_i = -1:
            ..., 2), 1), 0)
            ..., 2), 1)
            ...
            iterfunc(iterfunc(init_tensor, num_iters - 1), num_iters - 2)
            iterfunc(init_tensor, num_iters - 1)
            init_tensor
    For example:
    ```python
        unfold(
            init_tensor=tf.constant(0),
            iterfunc=lambda x, i: x + i,
            num_iters=5,
            d_i=1,
            element_shape=tf.TensorShape([]),
        )
        # -> [0, 0, 1, 3, 6, 10]
    ```

    Args:
        init_tensor:    tf.Tensor, of any shape that is the initial value of the iterations.
        iterfunc:       tf.Tensor, int -> tf.Tensor, that is the iteration function
                            from and onto the same shape as init_tensor
        num_iters:      tf.Tensor or static integer that is the number of iterations
        d_i:            either +1 or -1, where
                            +1 corresponds for the iterations from 0 to num_iters inclusive
                            -1 corresponds for the iterations from num_iters to 0 inclusive
        element_shape:  tf.TensorShape([]) that is the shape of init_tensor
        swap_memory:    the same as for tf.while_loop, argument
        name:           str, local tensor names scope

    Returns:            tf.Tensor, shape = [num_iters + 1] + init_tensor.shape
                        dtype the same as init_tensor
    """
    assert d_i in {-1, 1}
    positive_direction = d_i == 1

    with tf.name_scope(name):
        if isinstance(num_iters, int):
            num_iters = tf.constant(num_iters, dtype=tf.int32)
        elif isinstance(num_iters, tf.Tensor):
            pass
        else:
            raise ValueError(f"num_iters argument must be a pythonic integer o a tf.Tensor "
                             f"but it has type '{type(num_iters)}'.")

        tensor_array = tf.TensorArray(
            dtype=init_tensor.dtype,
            size=num_iters + 1,
            element_shape=element_shape,
            clear_after_read=False,
            infer_shape=True,
            dynamic_size=False,
        )
        tensor_array = tensor_array.write(0 if positive_direction else num_iters, init_tensor)

        def body(i, tensor_slice):
            last_value = tensor_slice.read(i if positive_direction else i + 1)
            new_value = iterfunc(last_value, i)
            tensor_slice = tensor_slice.write(i + 1 if positive_direction else i, new_value)
            return i + d_i, tensor_slice

        n = tf.constant(0, dtype=tf.int32) if positive_direction else num_iters - 1
        _, array_out = tf.while_loop(
            cond=lambda i, _: tf.constant(True),
            body=body,
            loop_vars=(n, tensor_array),
            maximum_iterations=num_iters,
            swap_memory=swap_memory,
            name=f"unfold_while_loop",
        )
        return array_out.stack()


def reduce_max_with_default(input_tensor: tf.Tensor, default: tf.Tensor) -> tf.Tensor:
    """A version of tf.reduce_max function that supports default values for zero size input.
    Support axis=None case only that corresponds to scalar output

    Args:
        input_tensor:   tf.Tensor, of any shape and numerical type
        default:        tf.Tensor, shape = [], dtype the same as input_tensor

    Returns:            tf.Tensor, shape = [], dtype the same as input_tensor
    """
    total_size = tf.shape(tf.reshape(input_tensor, [-1]))[0]
    return tf.where(
        condition=total_size > 0,
        x=tf.reduce_max(input_tensor),
        y=default
    )
