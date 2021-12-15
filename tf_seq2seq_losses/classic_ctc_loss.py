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
import numpy as np
from cached_property import cached_property
import tensorflow as tf

from tf_seq2seq_losses.base_loss import BaseCtcLossData, ctc_loss
from tf_seq2seq_losses.tools import logsumexp, apply_logarithmic_mask, unfold, expand_many_dims


def classic_ctc_loss(
        labels: tf.Tensor,
        logits: tf.Tensor,
        label_length: tf.Tensor,
        logit_length: tf.Tensor,
        blank_index: Union[int, tf.Tensor] = 0,
) -> tf.Tensor:
    """Computes CTC (Connectionist Temporal Classification) loss from
    http://www.cs.toronto.edu/~graves/icml_2006.pdf.

    Repeated non-blank labels will be merged.
    For example, predicted sequence
        a_bb_ccc_cc
    corresponds to label
        abcc
    where "_" is the blank token.

    If label length is longer then the logit length the output loss for the corresponding sample in the batch
    is +tf.inf and the gradient is 0. For example, for label "abb" at least 4 tokens are needed.
    It is because the output sequence must be at least "ab_b" in order to handle the repeated token.

    Args:
        labels:         tf.Tensor, shape = [batch, max_label_length],       dtype = tf.int32
        logits:         tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
        label_length:   tf.Tensor, shape = [batch],                         dtype = tf.int32
        logit_length:   tf.Tensor, shape = [batch],                         dtype = tf.int32
        blank_index:    tf.Tensor or pythonic static integer between 0 <= blank_index < mum_tokens

    Returns:            tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
    """
    return ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        blank_index=blank_index,
        ctc_loss_data_cls=ClassicCtcLossData
    )


class ClassicCtcLossData(BaseCtcLossData):
    """Calculate loss data for CTC (Connectionist Temporal Classification) loss from
    http://www.cs.toronto.edu/~graves/icml_2006.pdf.

    This loss is actually the logarithmic likelihood for the classification task with multiple expected class.
    All predicated sequences consist of tokens (denoted like "a", "b", ... below) and the blank "_".
    The classic CTC decoding merges all repeated non-blank labels and removes the blank.
    For example, predicted sequence
        a_bb_ccc_c is decoded as "abcc".
    All predicated sequences that coincided with the label after the decoding are the expected classes
    in the logarithmic likelihood loss function.

    Implementation:

    We calculate alpha_{b,t,l,s} and beta_{b,t,l,s} that are the logarithmic probabilities similar to
    this the ones from the sited paper and defined precisely below.
    Here, b corresponds to batch, t to logit position, l to label index, and s=0,1 to state (see below for details).

    During the decoding procedure, after handling of a part of the logit sequence,
    we predict only a part of the target label tokens. We call this subsequence the in the target space as "state".
    For example, two decode label "abc" we have to decode "a" first then add "b" and move tot the state "ab" and
    then to the state "abc".

    In order to handle the token duplication swap in the classic CTC loss we extend the set of all possible labels.
    For each token sequence we define two sequences called "closed" and "open".
    For example, for label "abc" we consider its two states denoted "abc>" (closed) and "abc<" (open).
    The difference between them is in their behaviour with respect to the token appending. The rules are:
        "...a>" + "_" -> "...a>",
        "...a<" + "_" -> "...a>",
        "...a>" + "a" -> "...aa<",
        "...a<" + "a" -> "...a<",
        "...a>" + "b" -> "...ab<",
        "...a<" + "b" -> "...ab<",
    for any different tokens "a" and "b" and any token sequence denoted by "...".
    Namely, appending a token the is equal to the last one to an open state does not change this state.
    Appending a blank to a state always males this state closed.

    This is why alpha_{b,t,l,s} and beta_{b,t,l,s} in the code below are equipped with an additional index s=0,1.
    Closed states corresponds s=0 and open ones to s=1.

    In particular, the flowing identity is satisfied
        sum_s sum_l exp alpha_{b,t,l,s} * exp beta_{b,t,l,s} = loss_{b}, for any b and t
    """
    @cached_property
    def diagonal_non_blank_grad_term(self) -> tf.Tensor:
        """ shape = [batch_size, max_logit_length, num_tokens] """
        input_tensor = \
            self.alpha[:, :-1] \
            + self.any_to_open_diagonal_step_log_proba \
            + tf.roll(self.beta[:, 1:, :, 1:], shift=-1, axis=2)
        # shape = [batch_size, max_logit_length, max_label_length + 1, states]
        act = tf.reduce_logsumexp(
            input_tensor=input_tensor,
            axis=3,
        )
        # shape = [batch_size, max_logit_length, max_label_length + 1]
        diagonal_non_blank_grad_term = self.select_from_act(act=act, label=self.label)
        # shape = [batch_size, max_logit_length, num_tokens]
        return diagonal_non_blank_grad_term

    @cached_property
    def horizontal_non_blank_grad_term(self) -> tf.Tensor:
        """Horizontal steps from repeated token: open alpha state to open beta state.

        Returns: shape = [batch_size, max_logit_length, num_tokens]
        """
        act = self.alpha[:, :-1, :, 1] + self.previous_label_token_log_proba + self.beta[:, 1:, :, 1]
        # shape = [batch_size, max_logit_length, max_label_length + 1]
        horizontal_non_blank_grad_term = self.select_from_act(act, self.preceded_label)
        return horizontal_non_blank_grad_term

    @cached_property
    def loss(self) -> tf.Tensor:
        """ shape = [batch_size] """
        params = tf.reduce_logsumexp(self.alpha[:, -1], -1)
        # shape = [batch_size, max_label_length + 1]
        loss = -tf.gather(
            params=params,                # shape = [batch_size, max_label_length + 1]
            indices=self.label_length,    # shape = [batch_size]
            batch_dims=1,
        )
        return loss

    @cached_property
    def gamma(self) -> tf.Tensor:
        """ shape = [
                batch_size,
                max_logit_length + 1,
                max_label_length + 1,
                state,
                max_logit_length + 1,
                max_label_length + 1,
                state,
            ],
        """
        # This is to avoid InaccessibleTensorError in graph mode
        _, _, _ = self.horizontal_step_log_proba, self.any_to_open_diagonal_step_log_proba, self.diagonal_gamma

        gamma_forward_transposed = unfold(
            init_tensor=self.diagonal_gamma,
            # init_tensor=tf.tile(self.diagonal_gamma, [self.batch_size, self.max_logit_length_plus_one, 1, 1, 1, 1]),
            iterfunc=self.gamma_step,
            d_i=1,
            num_iters=self.max_logit_length,
            element_shape=tf.TensorShape([None, None, None, None, None, None]),
            name="gamma_1",
        )
        # shape = [max_logit_length + 1, batch_size, max_logit_length + 1, max_label_length + 1, state,
        #   max_label_length + 1, state]

        gamma_forward = tf.transpose(gamma_forward_transposed, [1, 2, 3, 4, 0, 5, 6])
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state,
        #   max_logit_length + 1, max_label_length + 1, state]

        mask = expand_many_dims(
            input=tf.linalg.band_part(tf.ones(shape=[self.max_logit_length_plus_one] * 2, dtype=tf.bool), 0, -1),
            axes=[0, 2, 3, 5, 6]
        )
        # shape = [1, max_logit_length + 1, 1, 1, max_logit_length + 1, 1, 1]
        gamma = apply_logarithmic_mask(gamma_forward, mask)
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state,
        #   max_logit_length + 1, max_label_length + 1, state]

        return gamma

    def gamma_step(
        self,
        previous_slice: tf.Tensor,
        i: tf.Tensor,
    ) -> tf.Tensor:
        """Args:
            previous_slice: tf.Tensor,
                            shape = [batch_size, max_logit_length + 1, max_label_length + 1, state,
                                max_label_length + 1, state]
            i:              tf.Tensor,
                            shape = [], 0 <= i < max_logit_length + 1

        Returns:            tf.Tensor,
                            shape = [batch_size, max_logit_length + 1, max_label_length + 1, state,
                                max_label_length + 1, state]
        """
        horizontal_step_states = \
            expand_many_dims(self.horizontal_step_log_proba[:, i], axes=[1, 2, 3]) \
            + tf.expand_dims(previous_slice, 5)
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state,
        #          max_label_length + 1, next_state, previous_state]
        horizontal_step = tf.reduce_logsumexp(horizontal_step_states, axis=6)
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state, max_label_length + 1, state]

        diagonal_step_log_proba = tf.reduce_logsumexp(
            expand_many_dims(self.any_to_open_diagonal_step_log_proba[:, i], axes=[1, 2, 3]) + previous_slice,
            axis=5
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state, max_label_length + 1]

        # We move by one token because it is a diagonal step
        moved_diagonal_step_log_proba = tf.roll(diagonal_step_log_proba, shift=1, axis=4)
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state, max_label_length + 1]

        # Out state is always open:
        diagonal_step = tf.pad(
            tensor=tf.expand_dims(moved_diagonal_step_log_proba, 5),
            paddings=[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]],
            constant_values=-np.inf
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state, max_label_length + 1, state]
        new_gamma_slice = logsumexp(
            x=horizontal_step,
            y=diagonal_step,
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state, max_label_length + 1, state]

        condition = tf.reshape(tf.range(self.max_logit_length_plus_one) <= i, shape=[1, -1, 1, 1, 1, 1])
        # shape = [1, max_logit_length + 1, 1, 1, 1, 1, 1]
        output_slice = tf.where(
            condition=condition,
            x=new_gamma_slice,
            y=self.diagonal_gamma,
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state, max_label_length + 1, state]

        return output_slice

    @cached_property
    def diagonal_gamma(self) -> tf.Tensor:
        """ shape = [batch_size, max_logit_length_plus_one, max_label_length + 1, state,
                     max_label_length + 1, state]
        """
        diagonal_gamma = tf.math.log(
            tf.reshape(
                tensor=tf.eye(self.max_label_length_plus_one * 2, dtype=tf.float32),
                shape=[1, 1, self.max_label_length_plus_one, 2, self.max_label_length_plus_one, 2]
            )
        )
        diagonal_gamma = tf.tile(diagonal_gamma, [self.batch_size, self.max_logit_length_plus_one, 1, 1, 1, 1])
        return diagonal_gamma

    @cached_property
    def beta(self) -> tf.Tensor:
        """Calculates the beta_{b,t,l,s} that is logarithmic probability of sample 0 <= b < batch_size - 1 in the batch
        with logit subsequence from
            t, t + 1, ... max_logit_length - 2, max_logit_length - 1,
        for t < max_logit_length
        to predict the sequence of tokens
            w_max_label_length, w_{max_label_length + 1}, ... w_{max_label_length - 2}, w_{max_label_length - 1}
        for l < max_label_length
        that is either closed s=0 or open s=1.
        from label_b = [w_0, w_1, ... w_{max_label_length - 2}, w_{max_label_length - 1}].

        This logarithmic probability is calculated by iterations
            exp beta_{t-1,l} = p_horizontal_step_{t-1,l} * exp beta_{t,l} + p_diagonal_step_{t-1,l} * exp beta_{t,l+1},
        for 0 <= t < max_logit_length,
        where p_diagonal_step_{t,l} is the probability to predict label token w_l with logit l
        and p_horizontal_step_{t,l} is the probability to skip token w_l prediction with logit l, for example, with
        the blank prediction.

        Returns:    tf.Tensor,  shape = [batch_size, max_logit_length + 1, max_label_length + 1, state],
                    dtype = tf.float32
        """
        # This is to avoid InaccessibleTensorError in graph mode
        _, _ = self.horizontal_step_log_proba, self.any_to_open_diagonal_step_log_proba

        beta = unfold(
            init_tensor=self.last_beta_slice,
            iterfunc=self.beta_step,
            d_i=-1,
            num_iters=self.max_logit_length,
            element_shape=tf.TensorShape([None, None, 2]),
            name="beta",
        )
        # shape = [logit_length + 1, batch, label_length + 1, state]
        return tf.transpose(beta, [1, 0, 2, 3])

    def beta_step(self, previous_slice: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
        """ shape = [batch_size, max_label_length + 1, state] """
        horizontal_step = \
            tf.reduce_logsumexp(self.horizontal_step_log_proba[:, i] + tf.expand_dims(previous_slice, 3), 2)
        # shape = [batch_size, max_label_length + 1, state]
        diagonal_step = \
            self.any_to_open_diagonal_step_log_proba[:, i] + tf.roll(previous_slice[:, :, 1:], shift=-1, axis=1)
        # shape = [batch_size, max_label_length + 1, state]
        new_beta_slice = logsumexp(
            x=horizontal_step,  # shape = [batch_size, max_label_length + 1, state]
            y=diagonal_step,    # shape = [batch_size, max_label_length + 1, state]
        )
        # shape = [batch_size, max_label_length + 1, state]
        return new_beta_slice

    @cached_property
    def last_beta_slice(self) -> tf.Tensor:
        """ shape = [batch_size, max_label_length + 1, state] """
        beta_last = tf.math.log(tf.one_hot(indices=self.label_length, depth=self.max_label_length_plus_one))
        beta_last = tf.tile(input=tf.expand_dims(beta_last, axis=2), multiples=[1, 1, 2])
        return beta_last

    @cached_property
    def alpha(self) -> tf.Tensor:
        """Calculates the alpha_{b,t,l,s} that is
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

        Returns:    tf.Tensor,  shape = [batch_size, max_logit_length + 1, max_label_length + 1, state],
                    dtype = tf.float32
        """
        # This is to avoid InaccessibleTensorError in graph mode
        _, _ = self.horizontal_step_log_proba, self.any_to_open_diagonal_step_log_proba

        alpha = unfold(
            init_tensor=self.first_alpha_slice,
            iterfunc=self.alpha_step,
            d_i=1,
            num_iters=self.max_logit_length,
            element_shape=tf.TensorShape([None, None, 2]),
            name="alpha",
        )
        # shape = [logit_length + 1, batch_size, label_length + 1, state]
        return tf.transpose(alpha, [1, 0, 2, 3])

    def alpha_step(self, previous_slice: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
        """Args:
            previous_slice: shape = [batch_size, max_label_length + 1, state]
            i:

        Returns:            shape = [batch_size, max_label_length + 1, state]
        """
        temp = self.horizontal_step_log_proba[:, i] + tf.expand_dims(previous_slice, 2)
        # shape = [batch_size, max_label_length + 1, next_state, previous_state]
        horizontal_step = tf.reduce_logsumexp(temp, 3)
        # shape = [batch_size, max_label_length + 1, state]
        diagonal_step_log_proba = \
            tf.reduce_logsumexp(self.any_to_open_diagonal_step_log_proba[:, i] + previous_slice, 2)
        # shape = [batch_size, max_label_length + 1]

        # We move by one token because it is a diagonal step
        moved_diagonal_step_log_proba = tf.roll(diagonal_step_log_proba, shift=1, axis=1)
        # shape = [batch_size, max_label_length + 1]

        # Out state is always open:
        diagonal_step = tf.pad(
            tensor=tf.expand_dims(moved_diagonal_step_log_proba, 2),
            paddings=[[0, 0], [0, 0], [1, 0]],
            constant_values=-np.inf
        )
        # shape = [batch_size, max_label_length + 1, state]
        new_alpha_slice = logsumexp(
            x=horizontal_step,
            y=diagonal_step,
        )
        # shape = [batch_size, max_label_length + 1, state]
        return new_alpha_slice

    @cached_property
    def first_alpha_slice(self) -> tf.Tensor:
        """ shape = [batch_size, max_label_length + 1, state] """
        alpha_0 = tf.math.log(tf.one_hot(indices=0, depth=self.max_label_length_plus_one * 2))
        alpha_0 = tf.tile(input=tf.reshape(alpha_0, [1, -1, 2]), multiples=[self.batch_size, 1, 1])
        return alpha_0

    @cached_property
    def any_to_open_diagonal_step_log_proba(self) -> tf.Tensor:
        """Logarithmic probability to make a diagonal step from given state to an open state

        Returns:shape = [batch_size, max_logit_length, max_label_length + 1, state]
        """
        return tf.stack(
            values=[self.closed_to_open_diagonal_step_log_proba, self.open_to_open_diagonal_step_log_proba],
            axis=3
        )

    @cached_property
    def open_to_open_diagonal_step_log_proba(self) -> tf.Tensor:
        """Logarithmic probability to make a diagonal step from an open state to an open state
        with expected token prediction that is different from the previous one.

        Returns:shape = [batch_size, max_logit_length, max_label_length + 1]
        """
        # We check that the predicting token does not equal to previous one
        token_repetition_mask = self.label != tf.roll(self.label, shift=1, axis=1)
        # shape = [batch_size, max_label_length + 1]
        open_diagonal_step_log_proba = \
            apply_logarithmic_mask(
                self.closed_to_open_diagonal_step_log_proba,
                tf.expand_dims(token_repetition_mask, axis=1)
            )
        return open_diagonal_step_log_proba

    @cached_property
    def closed_to_open_diagonal_step_log_proba(self) -> tf.Tensor:
        """Logarithmic probability to make a diagonal step from a closed state to an open state
        with expected token prediction.

        Returns:shape = [batch_size, max_logit_length, max_label_length + 1]
        """
        return self.expected_token_logproba

    @cached_property
    def horizontal_step_log_proba(self) -> tf.Tensor:
        """Calculates logarithmic probability of the horizontal step for given logit x label position.

        This is possible in two alternative cases:
        1. Blank
        2. Not blank token from previous label position.

        Returns: tf.Tensor, shape = [batch_size, max_logit_length, max_label_length + 1, next_state, previous_state]
        """
        # We map closed and open states to closed states
        blank_term = tf.tile(
            input=tf.expand_dims(tf.expand_dims(self.blank_logproba, 2), 3),
            multiples=[1, 1, self.max_label_length_plus_one, 2]
        )
        # shape = [batch_size, max_logit_length, max_label_length + 1, 2]
        non_blank_term = tf.pad(
            tf.expand_dims(self.not_blank_horizontal_step_log_proba, 3),
            paddings=[[0, 0], [0, 0], [0, 0], [1, 0]],
            constant_values=tf.constant(-np.inf),
        )
        # shape = [batch_size, max_logit_length, max_label_length + 1, 2]
        horizontal_step_log_proba = tf.stack([blank_term, non_blank_term], axis=3)
        return horizontal_step_log_proba

    @cached_property
    def not_blank_horizontal_step_log_proba(self) -> tf.Tensor:
        """ shape = [batch_size, max_logit_length, max_label_length + 1] """
        mask = tf.reshape(1 - tf.one_hot(self.blank_token_index, depth=self.num_tokens), shape=[1, 1, -1])
        not_blank_log_proba = apply_logarithmic_mask(self.logproba, mask)
        not_blank_horizontal_step_log_proba = tf.gather(
            params=not_blank_log_proba,
            indices=tf.roll(self.label, shift=1, axis=1),
            axis=2,
            batch_dims=1,
        )
        # shape = [batch_size, max_logit_length, max_label_length + 1]
        return not_blank_horizontal_step_log_proba

    @cached_property
    def previous_label_token_log_proba(self) -> tf.Tensor:
        """Calculates the probability to predict token that preceded to label token.

        Returns:    tf.Tensor,  shape = [batch_size, max_logit_length, max_label_length + 1]
        """
        previous_label_token_log_proba = tf.gather(
            params=self.logproba,
            indices=self.preceded_label,
            axis=2,
            batch_dims=1,
        )
        # shape = [batch_size, max_logit_length, max_label_length + 1]
        return previous_label_token_log_proba

    @cached_property
    def blank_logproba(self) -> tf.Tensor:
        """ shape = [batch_size, max_logit_length] """
        return self.logproba[:, :, self.blank_token_index]

    def combine_transition_probabilities(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Transforms logarithmic transition probabilities a and b.

        Args:
            a:      shape = [batch, DIMS_A, max_logit_length, max_label_length + 1, state]
            b:      shape = [batch, max_logit_length, max_label_length + 1, state, DIMS_B]

        Returns:    shape = [batch, DIMS_A, max_logit_length, num_tokens, DIMS_B]
        """
        assert len(a.shape) >= 4
        assert len(b.shape) >= 4
        assert a.shape[-1] == 2
        assert b.shape[3] == 2

        dims_a = tf.shape(a)[1:-3]
        dims_b = tf.shape(b)[4:]
        a = tf.reshape(a, shape=[self.batch_size, -1, self.max_logit_length, self.max_label_length_plus_one, 2, 1])
        # shape = [batch_size, dims_a, max_logit_length, max_label_length + 1, state, 1]
        b = tf.reshape(b, shape=[self.batch_size, 1, self.max_logit_length, self.max_label_length_plus_one, 2, -1])
        # shape = [batch_size, 1, max_logit_length, max_label_length + 1, state, dims_b]

        # Either open or closed state from alpha and only closed state from beta
        ab_term = tf.reduce_logsumexp(a, 4) + b[:, :, :, :, 0]
        # shape = [batch_size, dims_a, max_logit_length, max_label_length + 1, dims_b]

        horizontal_blank_grad_term = \
            expand_many_dims(self.blank_logproba, axes=[1, 3]) + tf.reduce_logsumexp(ab_term, axis=3)
        # shape = [batch_size, dims_a, max_logit_length, dims_b]

        act = a[:, :, :, :, 1] + expand_many_dims(self.previous_label_token_log_proba, axes=[1, 4]) + b[:, :, :, :, 1]
        # shape = [batch_size, dim_a, max_logit_length, max_label_length + 1, dim_b]

        horizontal_non_blank_grad_term = self.select_from_act(act, self.preceded_label)
        # shape = [batch_size, dim_a, max_logit_length, num_tokens, dim_b]

        input_tensor = a + expand_many_dims(self.any_to_open_diagonal_step_log_proba, axes=[1, 5]) + \
            tf.roll(b[:, :, :, :, 1:], shift=-1, axis=3)
        # shape = [batch_size, dim_a, max_logit_length, max_label_length + 1, states, dim_b]

        act = tf.reduce_logsumexp(input_tensor=input_tensor, axis=4)
        # shape = [batch_size, dim_a, max_logit_length, max_label_length + 1, dim_b]

        diagonal_non_blank_grad_term = self.select_from_act(act=act, label=self.label)
        # shape = [batch_size, dim_a, max_logit_length, num_tokens, dim_b]

        non_blank_grad_term = logsumexp(horizontal_non_blank_grad_term, diagonal_non_blank_grad_term)
        # shape = [batch_size, dim_a, max_logit_length, num_tokens, dim_b]

        blank_mask = self.blank_token_index == tf.range(self.num_tokens)
        # shape = [num_tokens]

        output = tf.where(
            condition=expand_many_dims(blank_mask, axes=[0, 1, 2, 4]),
            x=tf.expand_dims(horizontal_blank_grad_term, 3),
            y=non_blank_grad_term,
        )
        # shape = [batch, dim_a, max_logit_length, num_tokens, dim_b]
        output_shape = tf.concat(
            [
                tf.expand_dims(self.batch_size, axis=0),
                dims_a,
                tf.expand_dims(self.max_logit_length, axis=0),
                tf.expand_dims(self.num_tokens, axis=0),
                dims_b
            ],
            axis=0
        )
        output_reshaped = tf.reshape(output, shape=output_shape)
        # shape = [batch, DIMS_A, max_logit_length, num_tokens, DIMS_B]

        return output_reshaped
