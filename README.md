# tf-seq2seq-losses
Tensorflow implementations for
[Connectionist Temporal Classification](file:///home/alexey/Downloads/Connectionist_temporal_classification_Labelling_un.pdf)
(CTC) loss that are fast and support second-order derivatives.

## Installation
```bash
$ pip install tf-seq2seq-losses
```

## Why Use This Package?
### 1. Faster Performance
Official CTC loss implementation, 
[`tf.nn.ctc_loss`](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss),
is significantly slower.
Our implementation is approximately 30 times faster, as shown by the benchmark results:

|       Name       | Forward Time (ms) | Gradient Calculation Time (ms) |                 
|:----------------:|:-----------------:|:------------------------------:|
|  `tf.nn.ctc_loss`  |    13.2 ± 0.02    |            10.4 ± 3            |
| `classic_ctc_loss` |   0.138 ± 0.006   |          0.28 ± 0.01           |
| `simple_ctc_loss`  |  0.0531 ± 0.003   |         0.119 ± 0.004          |

Tested on a single GPU: GeForce GTX 970, Driver Version: 460.91.03, CUDA Version: 11.2. For the experimental setup, see
[`benchmark.py`](tests/performance_test.py)
To reproduce this benchmark, run the following command from the project root directory 
(install `pytest` and `pandas` if needed):
```bash
$ pytest -o log_cli=true --log-level=INFO tests/benchmark.py
```
Here, classic_ctc_loss is the standard version of CTC loss with token collapsing, e.g., `a_bb_ccc_c -> abcc`. 
The simple_ctc_loss is a simplified version that removes blanks trivially, e.g., `a_bb_ccc_c -> abbcccc`.

### 2. Supports Second-Order Derivatives
This implementation supports second-order derivatives without using TensorFlow's autogradient. 
Instead, it uses a custom approach similar to the one described
[here](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss)
with a complexity of 
$O(l^4)$, 
where 
$l$
is the sequence length. The gradient complexity is 
$O(l^2)$.

Example usage:
```python
import tensorflow as tf
from tf_seq2seq_losses import classic_ctc_loss 

batch_size = 2
num_tokens = 3
logit_length = 5
labels = tf.constant([[1, 2, 2, 1], [1, 2, 1, 0]], dtype=tf.int32)
label_length = tf.constant([4, 3], dtype=tf.int32)
logits = tf.zeros(shape=[batch_size, logit_length, num_tokens], dtype=tf.float32)
logit_length = tf.constant([5, 4], dtype=tf.int32)

with tf.GradientTape(persistent=True) as tape1: 
    tape1.watch([logits])
    with tf.GradientTape() as tape2:
        tape2.watch([logits])
        loss = tf.reduce_sum(classic_ctc_loss(
            labels=labels,
            logits=logits,
            label_length=label_length,
            logit_length=logit_length,
            blank_index=0,
        ))
    gradient = tape2.gradient(loss, sources=logits)
hessian = tape1.batch_jacobian(gradient, source=logits, experimental_use_pfor=False)
# shape = [2, 5, 3, 5, 3]
```

### 3. Numerical Stability
1. The proposed implementation is more numerically stable, 
producing reasonable outputs even for logits of order `1e+10` and `-tf.inf`.
2. If the logit length is too short to predict the label output, 
the loss is `tf.inf` for that sample, unlike `tf.nn.ctc_loss`, which might output `707.13184`.


### 4. Pure Python Implementation
This is a pure Python/TensorFlow implementation, eliminating the need to build or compile any C++/CUDA components.


## Usage
The interface is identical to `tensorflow.nn.ctc_loss` with `logits_time_major=False`.

Example:
```python
import tensorflow as tf
from tf_seq2seq_losses import classic_ctc_loss

batch_size = 1
num_tokens = 3 # = 2 tokens + 1 blank token
logit_length = 5
loss = classic_ctc_loss(
    labels=tf.constant([[1, 2, 2, 1]], dtype=tf.int32),
    logits=tf.zeros(shape=[batch_size, logit_length, num_tokens], dtype=tf.float32),
    label_length=tf.constant([4], dtype=tf.int32),
    logit_length=tf.constant([logit_length], dtype=tf.int32),
    blank_index=0,
)
```

## Under the Hood
The implementation uses TensorFlow operations such as tf.while_loop and tf.TensorArray. 
The main computational bottleneck is the iteration over the logit length to calculate α and β 
(as described in the original
[CTC paper](file:///home/alexey/Downloads/Connectionist_temporal_classification_Labelling_un.pdf)). 
The expected gradient GPU calculation time is linear with respect to the logit length.

## Known Issues
### 1. Warning:
> AutoGraph could not transform <function classic_ctc_loss at ...> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 
(on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.

Observed with TensorFlow version 2.4.1. 
This warning does not affect performance and is caused by the use of Union in type annotations.

### 2. UnimplementedError:
Using `tf.jacobian` and `tf.batch_jacobian` for the second derivative of classic_ctc_loss with 
`experimental_use_pfor=False` in `tf.GradientTape` may cause an unexpected `UnimplementedError` 
in TensorFlow version 2.4.1 or later. 
This can be avoided by setting `experimental_use_pfor=True` 
or by using `ClassicCtcLossData.hessian` directly without `tf.GradientTape`.

Feel free to reach out if you have any questions or need further clarification.
