"""A set of auxiliary functions for numerical stability and tensor manipulations."""

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

from typing import Union, Callable, List, Optional
import tensorflow as tf
import numpy as np


inf = tf.constant(np.inf)


def logit_to_logproba(logit: tf.Tensor, axis: int) -> tf.Tensor:
    """Converts logits to logarithmic probabilities:
        logit_to_logproba(x) = x - log (sum along axis (exp(x)))

    Args:
        logit:  tf.Tensor, dtype = tf.float32
        axis: integer, like for tf.reduce_logsumexp

    Returns:    tf.Tensor, of the same shape and size as input logit
    """
    logprobas = logit - tf.reduce_logsumexp(
        input_tensor=logit, axis=axis, keepdims=True
    )
    return logprobas


def apply_logarithmic_mask(tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Masks a logarithmic representation of a tensor, namely
    1. Keeps the value of tensor unchanged for True values of mask
    2. Replace the value of tensor by -tf.inf for False values of mask

    Args:
        tensor: tf.Tensor, dtype = tf.float32 of the same shape as mask or broadcastable
        mask:   tf.Tensor, dtype = tf.bool of the same shape as tensor or broadcastable

    Returns:    tf.Tensor, dtype = tf.float32 of the same shape as tensor
    """
    return tensor + tf.math.log(tf.cast(mask, dtype=tf.float32))


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
        y=tf.where(condition=x > y, x=x + tf.math.softplus(y - x), y=x + np.log(2.0)),
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


def unsorted_segment_logsumexp(
    data: tf.Tensor, segment_ids: tf.Tensor, num_segments: Union[int, tf.Tensor]
) -> tf.Tensor:
    """Computes the logarithmic sum of exponents along segments of a tensor
    like other operators from tf.math.unsorted_segment_* family.

    Args:
        data:           tf.Tensor,  shape = [...] + data_dims,
        segment_ids:    tf.Tensor,  shape = [...], dtype = tf.int32
        num_segments:   tf.Tensor,  shape = [], dtype = tf.int32

    Returns:            tf.Tensor,  shape = [num_segments] + data_dims, for the same type as data
    """
    data_max = tf.math.unsorted_segment_max(
        data=data, segment_ids=segment_ids, num_segments=num_segments
    )
    data_normed = data - tf.gather(params=data_max, indices=segment_ids)
    output = data_max + tf.math.log(
        tf.math.unsorted_segment_sum(
            data=tf.exp(data_normed),
            segment_ids=segment_ids,
            num_segments=num_segments,
        )
    )
    return output


def pad_until(
    tensor: tf.Tensor,
    desired_size: Union[tf.Tensor, int],
    axis: int,
    pad_value: Union[tf.Tensor, int, float, bool] = 0,
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
    paddings = (
        [[0, 0]] * axis
        + [[0, desired_size - current_size]]
        + [[0, 0]] * (rank - axis - 1)
    )
    return tf.pad(tensor=tensor, paddings=paddings, constant_values=pad_value)


def insert_zeros(tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Inserts zeros into tensor before each masked element.
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
        tensor: tf.Tensor, shape = [batch, length], any type and the same shape as mask.
        mask:   tf.Tensor, shape = [batch, length], dtype = tf.bool and the same shape as tensor.

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
        shape=tf.stack([batch_size, length + max_num_insertions]),
    )

    return output


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
        iterfunc:       tf.Tensor, tf.Tensor -> tf.Tensor, that is the iteration function
                            from and onto the same shape as init_tensor
        num_iters:      tf.Tensor or static integer that is the number of iterations
        d_i:            either +1 or -1, where
                            +1 corresponds for the iterations from 0 to num_iters inclusive
                            -1 corresponds for the iterations from num_iters to 0 inclusive
        element_shape:  shape of init_tensor
        swap_memory:    the same as for tf.while_loop, argument
        name:           str, local tensor names scope

    Returns:            tf.Tensor, shape = [num_iters + 1] + init_tensor.shape
                        dtype the same as init_tensor
    """
    assert d_i in {-1, 1}
    positive_direction = d_i == 1

    with tf.name_scope(name):
        num_iters = tf.convert_to_tensor(num_iters)

        tensor_array = tf.TensorArray(
            dtype=init_tensor.dtype,
            size=num_iters + 1,
            element_shape=element_shape,
            clear_after_read=False,
            infer_shape=True,
            dynamic_size=False,
        )
        tensor_array = tensor_array.write(
            0 if positive_direction else num_iters, init_tensor
        )

        def body(i, tensor_slice):
            last_value = tensor_slice.read(i if positive_direction else i + 1)
            new_value = iterfunc(last_value, i)
            tensor_slice = tensor_slice.write(
                i + 1 if positive_direction else i, new_value
            )
            return i + d_i, tensor_slice

        n = tf.constant(0, dtype=tf.int32) if positive_direction else num_iters - 1
        _, array_out = tf.while_loop(
            cond=lambda i, _: tf.constant(True),
            body=body,
            loop_vars=(n, tensor_array),
            maximum_iterations=num_iters,
            swap_memory=swap_memory,
            name="unfold_while_loop",
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
    return tf.where(condition=total_size > 0, x=tf.reduce_max(input_tensor), y=default)


def expand_many_dims(x: tf.Tensor, axes: List[int]) -> tf.Tensor:
    """Analogous of tf.expand_dims for multiple new dimensions.
    Like for tf.expand_dims no new memory allocated for the output tensor.

    For example:
        expand_many_dims(tf.zeros(shape=[5, 1, 3]), axes=[0, 4, 5]).shape
        # -> [1, 5, 1, 3, 1, 1]

    Args:
        x:  tf.Tensor of any rank shape and type
        axes:   list of integer that are supposed to be the indexes of new dimensions.

    Returns:    tf.Tensor of the same type an input and rank = rank(input) + len(axes)
    """
    tensor = x
    for axis in axes:
        tensor = tf.expand_dims(input=tensor, axis=axis)

    return tensor


def smart_transpose(a: tf.Tensor, perm: List[int]) -> tf.Tensor:
    """Extension of tf.transpose.
    Parameter perm may be shorter list than rank on input tensor `a`.
    This case all dimensions that are beyond the list perm remain unchanged.

    For example:
        smart_transpose(tf.zeros(shape=[2, 3, 4, 5, 6]), [2, 1, 0]).shape
        # -> [4, 3, 2, 5, 6]

    Args:
        a:      tf.Tensor of any rank shape and type
        perm:   list of integers like for `tf.transpose` but in may be shorter than the shape of `a`.

    Returns:    tf.Tensor of the same type and rank as th input tensor `a`.
    """
    if len(perm) > len(a.shape):
        raise ValueError(
            f"Tensor with shape '{a.shape}' cannot be reshaped to '{perm}'"
        )
    perm_rest = list(range(len(perm), len(a.shape)))

    return tf.transpose(a=a, perm=perm + perm_rest)


def smart_reshape(
    tensor: tf.Tensor, shape: List[Optional[Union[int, tf.Tensor]]]
) -> tf.Tensor:
    """A version of tf.reshape.
    1. The output tensor is always of the same rank as input tensor.
    2. The parameter shape is supposed to be a list that is smaller or equal
    than the tensor shape.
    3. The list shape may contain None, that means "keep this dimension unchanged".
    4. The list shape is appended with None value to be of the same length as the input tensor shape.
    5. Like for `tf.reshape` output tensor does not requre new memory for allocation.

    For example:
    ```python
        smart_reshape(
            tensor=tf.zeros(shape=[2, 3, 4, 5]),
            shape=[8, None, 1]
        )
        # -> tf.Tensor([8, 3, 1, 5])
    ```

    Args:
        tensor: tf.Tensor of any shape and type
        shape:  list of optional static of dynamic integrates

    Returns:    tf.Tensor of the same typey and rank as the input tensor
    """
    if len(shape) > len(tensor.shape):
        raise ValueError(
            f"Tensor with shape {tensor.shape} cannot be reshaped to {shape}."
        )
    shape = shape + [None] * (len(tensor.shape) - len(shape))

    original_shape = tf.shape(tensor)
    new_shape = []
    for index, dim in enumerate(shape):
        if dim is None:
            new_shape.append(original_shape[index])
        else:
            new_shape.append(dim)

    return tf.reshape(tensor=tensor, shape=new_shape)
