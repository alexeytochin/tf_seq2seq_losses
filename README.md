# tf-seq2seq-losses
Tensorflow implementations for
[Connectionist Temporal Classification](file:///home/alexey/Downloads/Connectionist_temporal_classification_Labelling_un.pdf)
(CTC) loss in TensorFlow.

## Installation
Tested with Python 3.7. 
```bash
$ pip install tf-seq2seq-losses
```

## Why
### 1. Faster
Official CTC loss implementation 
[`tf.nn.ctc_loss`](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss)
is unreasonably slow. 
The proposed implementation is approximately 30 times faster as it follows form the benchmark:

|       name       |      forward time      |  gradient calculation time  |                 
|:----------------:|:----------------------:|:---------------------------:|
|  tf.nn.ctc_loss  |      13.2 ± 0.02       |          10.4 ± 3.          |
| classic_ctc_loss |     0.138 ± 0.006      |         0.28 ± 0.01         |
| simple_ctc_loss  |     0.0531 ± 0.003     |        0.119 ± 0.004        |

(Tested on single GPU: GeForce GTX 970,  Driver Version: 460.91.03, CUDA Version: 11.2). See 
[`benchmark.py`](tests/performance_test.py)
for the experimental setup. To reproduce this benchmark use
```bash
$ pytest -o log_cli=true --log-level=INFO tests/benchmark.py
```
from the project root directory.
Here `classic_ctc_loss` is the standard version of CTC loss
that corresponds to the decoding with repeated tokens collapse like 
> a_bb_ccc_c   ->   abcc

(equivalent to `tensorflow.nn.ctc_loss`).
The loss function `simple_ctc_loss` is a simplified version corresponding to the trivial decoding rule, for example,

> a_bb_ccc_c   ->   abbcccc

(simple blank removing).

### 2. Supports second order derivative
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
As well as for the first order gradient the TensorFlow autogradient is not used here.
Instead, an approach similar to 
[this one](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss)
is used with complexity 
<img src="https://render.githubusercontent.com/render/math?math=O(l^4)">
where 
<img src="https://render.githubusercontent.com/render/math?math=l">
is the sequence length (the complexity for the gradient is 
<img src="https://render.githubusercontent.com/render/math?math=O(l^2)">
). 

### 3. Numerically stable 
1. Proposed implementation is more numerically stable. For example, it calculates resonable output for
`logits` of order `1e+10` and even for `-tf.inf`.
2. If logit length is too short to predict label output the probability of expected prediction is zero.
Thus, the loss output is `tf.inf` for this sample but not `707.13184` for `tf.nn.ctc_loss`.


### 4. No C++ compilation
This is a pure Python/TensorFlow implementation. We do not have to build or compile any C++/CUDA stuff.


## Usage
The interface is identical to `tensorflow.nn.ctc_loss` with `logits_time_major=False`.
```python
import tensorflow as tf
from tf_seq2seq_losses import classic_ctc_loss

batch_size = 1
num_tokens = 3 # = 2 tokens + blank
logit_length = 5
loss = classic_ctc_loss(
    labels=tf.constant([[1, 2, 2, 1]], dtype=tf.int32),
    logits=tf.zeros(shape=[batch_size, logit_length, num_tokens], dtype=tf.float32),
    label_length=tf.constant([4], dtype=tf.int32),
    logit_length=tf.constant([logit_length], dtype=tf.int32),
    blank_index=0,
)
```

## Under the roof
TensorFlow operations sich as `tf.while_loop` and `tf.TensorArray`. 
The bottleneck is the iterations over the logit length in order to calculate
α and β
(see, for example, the original 
[paper](file:///home/alexey/Downloads/Connectionist_temporal_classification_Labelling_un.pdf)
). Expected gradient GPU calculation time is linear over logit length. 

## Known Problems
1. Warning:
> AutoGraph could not transform <function classic_ctc_loss at ...> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.

observed for TensorFlow version 2.4.1
has no effect for performance. It is caused by `Union` in type annotations. 

2. `tf.jacobian` and `tf.batch_jacobian` for second derivative of `classic_ctc_loss` 
with `experimental_use_pfor=False` in `tf.GradientTape` causes an unexpected `UnimplementedError`.
for TensorFlow version 2.4.1. This can be avoided by `experimental_use_pfor=True` 
or by using `ClassicCtcLossData.hessain` directly without `tf.GradientTape`. 

## Future plans
1. Add decoding (inference)
2. Add rnn-t loss.
3. Add m-wer loss.

