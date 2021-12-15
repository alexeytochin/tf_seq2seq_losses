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

import tensorflow as tf
from typing import Optional, Dict, Union

from tf_seq2seq_losses.tools import logit_to_logproba


def tf_ctc_loss(
        labels: tf.Tensor,
        logits: tf.Tensor,
        label_length: tf.Tensor,
        logit_length: tf.Tensor,
        blank_index: Union[int, tf.Tensor]=0,
):
    return tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        blank_index=None if not blank_index else blank_index,
        logits_time_major=False,
    )


def generate_ctc_loss_inputs(
        batch_size: int,
        max_logit_length: int,
        random_seed: Optional[int],
        num_tokens: int,
        blank_index: int,
) -> Dict[str, Union[tf.Tensor, int]]:
    """Generates random data for ctc-loss

    Args:
        batch_size:         batch size
        max_logit_length:   maximal logit length
        random_seed:        random seed
        num_tokens:         number of tokens
        blank_index:        blank index

    Returns:                dictionary with random tensors.
    """
    assert blank_index == 0
    if random_seed is not None:
        tf.random.set_seed(random_seed)
    logits = tf.random.normal(shape=[batch_size, max_logit_length, num_tokens], stddev=1)
    logit_length = tf.random.uniform(
        minval=max_logit_length // 2,
        maxval=max_logit_length,
        shape=[batch_size],
        dtype=tf.int32
    )
    label_length = tf.random.uniform(
        minval=max_logit_length // 4,
        maxval=max_logit_length // 2,
        shape=[batch_size],
        dtype=tf.int32
    )
    labels = tf.random.uniform(minval=1, maxval=num_tokens, shape=[batch_size, max_logit_length], dtype=tf.int32)
    log_probas = logit_to_logproba(logit=logits, axis=2)

    return {
        "labels": labels,
        "logits": logits,
        "logprobas": log_probas,
        "label_length": label_length,
        "logit_length": logit_length,
        "blank_index": blank_index
    }
