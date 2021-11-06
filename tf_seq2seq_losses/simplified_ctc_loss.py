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

from typing import Union
import tensorflow as tf
from cached_property import cached_property

from tf_seq2seq_losses.base_loss import BaseCtcLossData, ctc_loss
from tf_seq2seq_losses.tools import unfold, logsumexp


def simplified_ctc_loss(
        labels: tf.Tensor,
        logits: tf.Tensor,
        label_length: tf.Tensor,
        logit_length: tf.Tensor,
        blank_index: Union[int, tf.Tensor] = 0,
) -> tf.Tensor:
    """Computes a simpified version of CTC (Connectionist Temporal Classification) loss from
    http://www.cs.toronto.edu/~graves/icml_2006.pdf
    without the token swap feature. Thus, we simply remove the blank token during the decoding.
    For example, predicted sequence
        "a_bb_ccc_cc"
    corresponds to label
        "abbccccc",
    where "_" is the blank.

    If the label length is longer then the logit length the output loss for the corresponding sample in the batch
    is +tf.inf and the gradient is 0. For example, for label "aabc" at least 4 tokens are needed.

    Args:
        labels:         tf.Tensor, shape = [batch, max_label_length],       dtype = tf.int32
        logits:         tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
        label_length:   tf.Tensor, shape = [batch],                         dtype = tf.int32
        logit_length:   tf.Tensor, shape = [batch],                         dtype = tf.int32
        blank_index:    tf.Tensor or pythonic static integer, 0 <= blank_index < mum_tokens

    Returns:            tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
    """
    return ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        blank_index=blank_index,
        ctc_loss_data_cls=SimplifiedCtcLossData
    )


class SimplifiedCtcLossData(BaseCtcLossData):
    @cached_property
    def log_loss(self) -> tf.Tensor:
        """ shape = [batch_size] """
        params = self.alpha[:, -1]
        # shape = [batch_size, max_label_length + 1]
        loss = -tf.gather(
            params=params,                # shape = [batch_size, max_label_length + 1]
            indices=self.label_length,    # shape = [batch_size]
            batch_dims=1,
        )
        return loss

    @cached_property
    def non_blank_grad_term(self) -> tf.Tensor:
        """Calculates gradient term log [sum exp(alpha) * exp(log_proba) * exp(beta)]
        for non-blank tokens.

        Returns: tf.Tensor, shape = [batch_size, max_logit_length, num_tokens]
        The values at [:, :, self._token_index] are to be ignored
        """
        act = self.alpha[:, :-1] + self.expected_token_log_proba + tf.roll(self.beta[:, 1:], shift=-1, axis=2)
        # shape = [batch_size, max_logit_length, max_label_length + 1]
        diagonal_non_blank_grad_term = self.select_from_act(act=act, label=self.label)
        # shape = [batch_size, max_logit_length, num_tokens]
        return diagonal_non_blank_grad_term

    @cached_property
    def blank_grad_term(self) -> tf.Tensor:
        """Calculates gradient term log [sum exp(alpha) * exp(log_proba) * exp(beta)] for the blank token.

        Returns: tf.Tensor, shape = [batch_size, max_logit_length]
        """
        alpha_beta_term = self.alpha[:, :-1] + self.beta[:, 1:]
        # shape = [batch_size, logit_length, max_lobel_length + 1]
        horizontal_blank_grad_term = self.blank_log_proba + tf.reduce_logsumexp(alpha_beta_term, axis=2)
        # shape = [batch_size, logit_length]
        return horizontal_blank_grad_term

    @cached_property
    def beta(self) -> tf.Tensor:
        """Calculates the beta_{b,t,l} that is logarithmic probability of sample 0 <= b < batch_size - 1 in the batch
        with logit subsequence from
            t, t + 1, ... max_logit_length - 2, max_logit_length - 1,
        for t < max_logit_length
        to predict the sequence of tokens
            w_max_label_length, w_{max_label_length + 1}, ... w_{max_label_length - 2}, w_{max_label_length - 1}
        for l < max_label_length.
        from label_b = [w_0, w_1, ... w_{max_label_length - 2}, w_{max_label_length - 1}].

        This logarithmic probability is calculated by iterations
            exp beta_{t-1,l} = p_horizontal_step_{t-1,l} * exp beta_{t,l} + p_diagonal_step_{t-1,l} * exp beta_{t,l+1},
        for 0 <= t < max_logit_length,
        where p_diagonal_step_{t,l} is the probability to predict label token w_l with logit l
        and p_horizontal_step_{t,l} is the probability to skip token w_l prediction with logit l, for example, with
        the blank prediction.

        Returns:    tf.Tensor,  shape = [batch_size, max_logit_length + 1, max_label_length + 1],
                    dtype = tf.float32
        """
        horizontal_step_log_proba = self.horizontal_step_log_proba
        diagonal_step_log_proba = self.diagonal_step_log_proba

        def beta_step(previous_slice: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
            horizontal_step = tf.expand_dims(horizontal_step_log_proba[:, i], axis=1) + previous_slice
            # shape = [batch_size, max_label_length + 1]
            diagonal_step = diagonal_step_log_proba[:, i] + tf.roll(previous_slice, shift=-1, axis=1)
            # shape = [batch_size, max_label_length + 1]
            new_beta_slice = logsumexp(
                x=horizontal_step,  # shape = [batch_size, max_label_length + 1]
                y=diagonal_step,    # shape = [batch_size, max_label_length + 1]
            )
            # shape = [batch_size, max_label_length + 1]
            return new_beta_slice

        beta = unfold(
            init_tensor=self.last_beta_slice,
            iterfunc=beta_step,
            d_i=-1,
            num_iters=self.max_logit_length,
            element_shape=tf.TensorShape([None, None]),
            name="beta",
        )
        # shape = [logit_length + 1, batch, label_length + 1]
        return tf.transpose(beta, [1, 0, 2])

    @cached_property
    def last_beta_slice(self) -> tf.Tensor:
        """ shape = [batch_size, max_label_length + 1] """
        beta_last = tf.math.log(tf.one_hot(indices=self.label_length, depth=self.max_label_length + 1))
        return beta_last

    @cached_property
    def alpha(self) -> tf.Tensor:
        """Calculates the alpha_{b,t,l} that is
        the logarithmic probability of sample 0 <= b < batch_size - 1 in the batch
        with logits subsequence from 0, 1, 2, ... t - 2, t - 1, for t < max_logit_length
        to predict the sequence of tokens w_0, w_1, w_2, ... w_{l-2}, w_{l-1} for l < max_label_length + 1
        that is either closed s=0 or open s=1.
        from label_b = [w_0, w_1, ... w_{max_label_length - 2}, w_{max_label_length - 1}].

        This logarithmic probability is calculated by iterations
            exp alpha_{t + 1,l} = p_horizontal_step_{t,l} * exp alpha_{t,l} + p_diagonal_step_{t,l} * exp alpha_{t,l-1},
        for 0 <= t < max_logit_length,
        where p_diagonal_step_{t,l} is the probability to predict label token w_l with logit l
        and p_horizontal_step_{t,l} is the probability to skip token w_l prediction with logit l, for example, with
        the blank prediction.

        Returns:    tf.Tensor,  shape = [batch_size, max_logit_length + 1, max_label_length + 1],
                    dtype = tf.float32
        """
        horizontal_step_log_proba = self.horizontal_step_log_proba
        diagonal_step_log_proba = self.diagonal_step_log_proba

        def alpha_step(previous_slice: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
            horizontal_step = tf.expand_dims(horizontal_step_log_proba[:, i], axis=1) + previous_slice
            # shape = [batch_size, max_label_length + 1]
            diagonal_step = diagonal_step_log_proba[:, i] + previous_slice
            # shape = [batch_size, max_label_length + 1]
            # We move by one token because it is a diagonal step
            moved_diagonal_step = tf.roll(diagonal_step, shift=1, axis=1)
            # shape = [batch_size, max_label_length + 1]
            # Out state is always open:
            new_alpha_slice = logsumexp(
                x=horizontal_step,
                y=moved_diagonal_step,
            )
            # shape = [batch_size, max_label_length + 1]
            return new_alpha_slice

        alpha = unfold(
            init_tensor=self.first_alpha_slice,
            iterfunc=alpha_step,
            d_i=1,
            num_iters=self.max_logit_length,
            element_shape=tf.TensorShape([None, None]),
            name="alpha",
        )
        # shape = [logit_length + 1, batch_size, label_length + 1]
        return tf.transpose(alpha, [1, 0, 2])

    @cached_property
    def first_alpha_slice(self) -> tf.Tensor:
        """ shape = [batch_size, max_label_length + 1] """
        alpha_0 = tf.math.log(tf.one_hot(indices=0, depth=(self.max_label_length + 1)))
        alpha_0 = tf.tile(input=tf.expand_dims(alpha_0, 0), multiples=[self.batch_size, 1])
        return alpha_0

    @cached_property
    def horizontal_step_log_proba(self) -> tf.Tensor:
        """Logarithmic probability for the horizontal step for given logit x label position.

        Returns:    tf.Tensor,  shape = [batch_size, max_logit_length]
        """
        return self.log_proba[:, :, self.blank_token_index]

    @cached_property
    def diagonal_step_log_proba(self) -> tf.Tensor:
        """Logarithmic probability to make a diagonal step with expected token prediction.

        Returns:    tf.Tensor,  shape = [batch_size, max_logit_length, max_label_length + 1]
        """
        return self.expected_token_log_proba
