"""Simplified version of CTC (Connectionist Temporal Classification) loss."""

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

from typing import Union
from functools import cached_property

import tensorflow as tf
from tf_seq2seq_losses.base_loss import BaseCtcLossData, ctc_loss
from tf_seq2seq_losses.tools import (
    unfold,
    logsumexp,
    expand_many_dims,
    apply_logarithmic_mask,
)


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
        ctc_loss_data_cls=SimplifiedCtcLossData,
    )


class SimplifiedCtcLossData(BaseCtcLossData):
    """Data class for simplified CTC loss."""

    @cached_property
    def loss(self) -> tf.Tensor:
        """shape = [batch_size]"""
        params = self.alpha[:, -1]
        # shape = [batch_size, max_label_length + 1]
        loss = -tf.gather(
            params=params,  # shape = [batch_size, max_label_length + 1]
            indices=self._label_length,  # shape = [batch_size]
            batch_dims=1,
        )
        return loss

    @cached_property
    def gamma(self) -> tf.Tensor:
        """Transition logarithmic probability

        Returns:    tf.Tensor,
                    shape = [
                        batch_size,
                        max_logit_length + 1,
                        max_label_length + 1,
                        max_logit_length + 1,
                        max_label_length + 1,
                    ],
                    dtype = tf.float32
        """
        # This is to avoid InaccessibleTensorError in graph mode
        _, _, _ = (
            self.horizontal_step_log_proba,
            self.diagonal_step_log_proba,
            self.diagonal_gamma,
        )

        init_tensor = tf.math.log(
            tf.tile(
                tf.reshape(
                    tf.eye(self._max_label_length_plus_one),
                    [
                        1,
                        1,
                        self._max_label_length_plus_one,
                        self._max_label_length_plus_one,
                    ],
                ),
                [self._batch_size, self._max_logit_length_plus_one, 1, 1],
            )
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, max_label_length + 1]
        gamma_forward_transposed = unfold(
            init_tensor=init_tensor,
            iterfunc=self.gamma_step,
            d_i=1,
            num_iters=self._max_logit_length,
            element_shape=tf.TensorShape([None, None, None, None]),
            name="gamma",
        )
        # shape = [max_logit_length + 1, batch_size, max_logit_length + 1, max_label_length + 1, max_label_length + 1]

        gamma_forward = tf.transpose(gamma_forward_transposed, [1, 2, 3, 0, 4])
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, max_logit_length + 1, max_label_length + 1]
        mask = expand_many_dims(
            x=tf.linalg.band_part(
                tf.ones(shape=[self._max_logit_length_plus_one] * 2, dtype=tf.bool),
                0,
                -1,
            ),
            axes=[0, 2, 4],
        )
        gamma = apply_logarithmic_mask(gamma_forward, mask)
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, max_logit_length + 1, max_label_length + 1]

        return gamma

    def gamma_step(
        self,
        previous_slice: tf.Tensor,
        i: tf.Tensor,
    ) -> tf.Tensor:
        """Iteration step for gamma computing
        Args:
            previous_slice: tf.Tensor,
                            shape = [batch_size, max_logit_length + 1, max_label_length + 1, max_label_length + 1]
            i:              tf.Tensor, shape = [], 0 <= i < max_logit_length + 1

        Returns:            tf.Tensor,
                            shape = [batch_size, max_logit_length + 1, max_label_length + 1, max_label_length + 1]
        """
        horizontal_step = (
            expand_many_dims(self.horizontal_step_log_proba[:, i], axes=[1, 2, 3])
            + previous_slice
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, max_label_length + 1]
        diagonal_step = (
            expand_many_dims(self.diagonal_step_log_proba[:, i], axes=[1, 2])
            + previous_slice
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, max_label_length + 1]

        # We move by one token because it is a diagonal step
        moved_diagonal_step = tf.roll(diagonal_step, shift=1, axis=3)
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, max_label_length + 1]
        new_alpha_slice = logsumexp(
            x=horizontal_step,
            y=moved_diagonal_step,
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, max_label_length + 1]
        condition = tf.reshape(
            tf.range(self._max_logit_length_plus_one) <= i, shape=[1, -1, 1, 1]
        )
        # shape = [1, max_logit_length + 1, 1, 1]

        output_slice = tf.where(
            condition=condition,
            x=new_alpha_slice,
            y=self.diagonal_gamma,
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, max_label_length + 1]

        return output_slice

    @cached_property
    def last_gamma_slice(self) -> tf.Tensor:
        """shape = [batch_size, max_label_length + 1, logit_length + 1, label_length + 1]"""
        gamma_0 = tf.math.log(tf.eye(self._max_label_length_plus_one))
        # shape = [max_label_length + 1, max_label_length + 1]
        gamma_0 = tf.tile(
            input=tf.expand_dims(gamma_0, 0), multiples=[self._batch_size, 1, 1]
        )
        # shape = [batch_size, max_label_length + 1, max_label_length + 1]
        gamma_0 = tf.reshape(
            tensor=gamma_0,
            shape=[
                self._batch_size * self._max_label_length_plus_one,
                self._max_label_length_plus_one,
            ],
        )
        # shape = [batch_size * (max_label_length + 1), max_label_length + 1]

        last_gamma_slice = unfold(
            init_tensor=gamma_0,
            iterfunc=self.beta_step,
            d_i=-1,
            num_iters=self._max_logit_length,
            element_shape=tf.TensorShape([None, None]),
            name="last_gamma_slice_unfold",
        )
        # shape = [logit_length + 1, batch_size * (max_label_length + 1), label_length + 1]

        last_gamma_slice = tf.transpose(last_gamma_slice, perm=[1, 0, 2])
        # shape = [batch_size * (max_label_length + 1), logit_length + 1, label_length + 1]
        last_gamma_slice = tf.reshape(
            last_gamma_slice,
            shape=[
                self._batch_size,
                self._max_label_length_plus_one,
                self._max_logit_length_plus_one,
                self._max_label_length_plus_one,
            ],
        )
        # shape = [batch_size, max_label_length + 1, logit_length + 1, label_length + 1]

        return last_gamma_slice

    @cached_property
    def first_gamma_slice(self) -> tf.Tensor:
        """shape = [batch_size, max_label_length + 1, logit_length + 1, label_length + 1]"""
        gamma_0 = tf.math.log(tf.eye(self._max_label_length_plus_one))
        # shape = [max_label_length + 1, max_label_length + 1]
        gamma_0 = tf.tile(
            input=tf.expand_dims(gamma_0, 0), multiples=[self._batch_size, 1, 1]
        )
        # shape = [batch_size, max_label_length + 1, max_label_length + 1]
        gamma_0 = tf.reshape(
            gamma_0,
            shape=[
                self._batch_size * self._max_label_length_plus_one,
                self._max_label_length_plus_one,
            ],
        )
        # shape = [batch_size * (max_label_length + 1), max_label_length + 1]

        first_gamma_slice = unfold(
            init_tensor=gamma_0,
            iterfunc=self.alpha_step,
            d_i=1,
            num_iters=self._max_logit_length,
            element_shape=tf.TensorShape([None, None]),
            name="first_gamma_slice_unfold",
        )
        # shape = [logit_length + 1, batch_size * (max_label_length + 1), label_length + 1]

        first_gamma_slice = tf.transpose(first_gamma_slice, perm=[1, 0, 2])
        # shape = [batch_size * (max_label_length + 1), logit_length + 1, label_length + 1]
        first_gamma_slice = tf.reshape(
            first_gamma_slice,
            shape=[
                self._batch_size,
                self._max_label_length_plus_one,
                self._max_logit_length_plus_one,
                self._max_label_length_plus_one,
            ],
        )
        # shape = [batch_size, max_label_length + 1, logit_length + 1, label_length + 1]

        return first_gamma_slice

    @cached_property
    def diagonal_gamma(self) -> tf.Tensor:
        """shape = [1, 1, max_label_length + 1, max_label_length + 1]"""
        return tf.math.log(
            tf.expand_dims(
                tf.expand_dims(
                    tf.eye(self._max_label_length_plus_one, dtype=tf.float32), axis=0
                ),
                axis=0,
            )
        )

    @cached_property
    def beta(self) -> tf.Tensor:
        """Calculates the beta_{b,t,l} that is logarithmic probability of sample number
        0 <= b < batch_size - 1
        in the batch with logit subsequence from
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
        # This is to avoid InaccessibleTensorError in graph mode
        _, _ = self.horizontal_step_log_proba, self.diagonal_step_log_proba

        beta_transposed = unfold(
            init_tensor=self.last_beta_slice,
            iterfunc=self.beta_step,
            d_i=-1,
            num_iters=self._max_logit_length,
            element_shape=tf.TensorShape([None, None]),
            name="beta",
        )
        # shape = [logit_length + 1, batch, label_length + 1]
        return tf.transpose(beta_transposed, [1, 0, 2])

    def beta_step(self, previous_slice: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
        """Iteration step for beta computation."""
        horizontal_step = (
            tf.expand_dims(self.horizontal_step_log_proba[:, i], axis=1)
            + previous_slice
        )
        # shape = [batch_size, max_label_length + 1]
        diagonal_step = self.diagonal_step_log_proba[:, i] + tf.roll(
            previous_slice, shift=-1, axis=1
        )
        # shape = [batch_size, max_label_length + 1]
        new_beta_slice = logsumexp(
            x=horizontal_step,  # shape = [batch_size, max_label_length + 1]
            y=diagonal_step,  # shape = [batch_size, max_label_length + 1]
        )
        # shape = [batch_size, max_label_length + 1]
        return new_beta_slice

    @cached_property
    def last_beta_slice(self) -> tf.Tensor:
        """Last beta slice for beta computation.

        Returns: shape = [batch_size, max_label_length + 1]
        """
        beta_last = tf.math.log(
            tf.one_hot(
                indices=self._label_length, depth=self._max_label_length_plus_one
            )
        )
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

        Returns:    tf.Tensor, shape = [batch_size, max_logit_length + 1, max_label_length + 1],
                    dtype = tf.float32
        """

        # This is to avoid InaccessibleTensorError in graph mode
        _, _ = self.horizontal_step_log_proba, self.diagonal_step_log_proba

        alpha_transposed = unfold(
            init_tensor=self.first_alpha_slice,
            iterfunc=self.alpha_step,
            d_i=1,
            num_iters=self._max_logit_length,
            element_shape=tf.TensorShape([None, None]),
            name="alpha",
        )
        # shape = [logit_length + 1, batch_size, label_length + 1]

        return tf.transpose(alpha_transposed, [1, 0, 2])

    def alpha_step(
        self,
        previous_slice: tf.Tensor,
        i: tf.Tensor,
    ) -> tf.Tensor:
        """Iteration step for alpha computation

        Args:
            previous_slice: shape = [batch_size] + DIMS + [max_label_length + 1]
            i:              shape = []
        Returns:            shape = [batch_size] + DIMS + [max_label_length + 1]
        """
        horizontal_step = (
            tf.expand_dims(self.horizontal_step_log_proba[:, i], axis=1)
            + previous_slice
        )
        # shape = [batch_size, max_label_length + 1]
        diagonal_step = self.diagonal_step_log_proba[:, i] + previous_slice
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

    @cached_property
    def first_alpha_slice(self) -> tf.Tensor:
        """First alpha slice for alpha computation.

        Returns: shape = [batch_size, max_label_length + 1]
        """
        alpha_0 = tf.math.log(
            tf.one_hot(indices=0, depth=self._max_label_length_plus_one)
        )
        alpha_0 = tf.tile(
            input=tf.expand_dims(alpha_0, 0), multiples=[self._batch_size, 1]
        )
        return alpha_0

    @cached_property
    def horizontal_step_log_proba(self) -> tf.Tensor:
        """Logarithmic probability for the horizontal step for given logit x label position.

        Returns:    tf.Tensor,  shape = [batch_size, max_logit_length]
        """
        return self._logproba[:, :, self._blank_token_index]

    @cached_property
    def diagonal_step_log_proba(self) -> tf.Tensor:
        """Logarithmic probability to make a diagonal step with expected token prediction.

        Returns:    tf.Tensor,  shape = [batch_size, max_logit_length, max_label_length + 1]
        """
        return self._expected_token_logproba

    def _combine_transition_probabilities(
        self, a: tf.Tensor, b: tf.Tensor
    ) -> tf.Tensor:
        """Combines transition logarithmic probabilities.

        Args:
            a:      shape = [batch, DIMS_A, max_logit_length, max_label_length + 1]
            b:      shape = [batch, max_logit_length, max_label_length + 1, DIMS_B]

        Returns:    shape = [batch, DIMS_A, max_logit_length, num_tokens, DIMS_B]
        """
        assert len(a.shape) >= 3
        assert len(b.shape) >= 3

        dims_a = tf.shape(a)[1:-2]
        dims_b = tf.shape(b)[3:]
        a = tf.reshape(
            a,
            shape=[
                self._batch_size,
                -1,
                self._max_logit_length,
                self._max_label_length_plus_one,
                1,
            ],
        )
        # shape = [batch_size, dims_a, max_logit_length, max_label_length + 1, 1]
        b = tf.reshape(
            b,
            shape=[
                self._batch_size,
                1,
                self._max_logit_length,
                self._max_label_length_plus_one,
                -1,
            ],
        )
        # shape = [batch_size, 1, max_logit_length, max_label_length + 1, dims_b]

        ab_term = a + b
        # shape = [batch_size, dims_a, max_logit_length, max_label_length + 1, dims_b]

        horizontal_blank_grad_term = expand_many_dims(
            self._blank_logproba, axes=[1, 3]
        ) + tf.reduce_logsumexp(ab_term, axis=3)
        # shape = [batch_size, dims_a, max_logit_length, dims_b]

        act = (
            a
            + expand_many_dims(self._expected_token_logproba, [1, 4])
            + tf.roll(b, shift=-1, axis=3)
        )
        # shape = [batch_size, dims_a, max_logit_length, max_label_length + 1, dims_b]

        diagonal_non_blank_grad_term = self._select_from_act(act=act, label=self._label)
        # shape = [batch_size, dims_a, max_logit_length, num_tokens, dims_b]

        blank_mask = self._blank_token_index == tf.range(self._num_tokens)
        # shape = [num_tokens]

        output = tf.where(
            condition=tf.reshape(blank_mask, [1, 1, 1, -1, 1]),
            x=tf.expand_dims(horizontal_blank_grad_term, 3),
            y=diagonal_non_blank_grad_term,
        )
        # shape = [batch_size, dims_a, max_logit_length, num_tokens, dims_b]
        output_shape = tf.concat(
            [
                tf.expand_dims(self._batch_size, axis=0),
                dims_a,
                tf.expand_dims(self._max_logit_length, axis=0),
                tf.expand_dims(self._num_tokens, axis=0),
                dims_b,
            ],
            axis=0,
        )
        output_reshaped = tf.reshape(output, shape=output_shape)

        return output_reshaped
