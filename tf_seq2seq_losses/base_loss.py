"""Base class for CTC loss data."""

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

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, Type
from functools import cached_property
import tensorflow as tf

from tf_seq2seq_losses.tools import (
    logit_to_logproba,
    pad_until,
    reduce_max_with_default,
    unsorted_segment_logsumexp,
    apply_logarithmic_mask,
    smart_transpose,
    smart_reshape,
    expand_many_dims,
    inf,
)


def ctc_loss(
    labels: tf.Tensor,
    logits: tf.Tensor,
    label_length: tf.Tensor,
    logit_length: tf.Tensor,
    blank_index: Union[int, tf.Tensor],
    ctc_loss_data_cls: Type[BaseCtcLossData],
) -> tf.Tensor:
    """Computes a version of CTC loss from
    http://www.cs.toronto.edu/~graves/icml_2006.pdf.

    Args:
        labels:             tf.Tensor, shape = [batch, max_label_length],       dtype = tf.int32
        logits:             tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
        label_length:       tf.Tensor, shape = [batch],                         dtype = tf.int32
        logit_length:       tf.Tensor, shape = [batch],                         dtype = tf.int32
        blank_index:        static integer >= 0
        ctc_loss_data_cls:  BaseCtcLossData class

    Returns:                tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
    """
    log_probas = logit_to_logproba(logit=logits, axis=2)
    loss = ctc_loss_from_logproba(
        labels=labels,
        logprobas=log_probas,
        label_length=label_length,
        logit_length=logit_length,
        blank_index=blank_index,
        ctc_loss_data_cls=ctc_loss_data_cls,
    )
    return loss


def ctc_loss_from_logproba(
    labels: tf.Tensor,
    logprobas: tf.Tensor,
    label_length: tf.Tensor,
    logit_length: tf.Tensor,
    blank_index: Union[int, tf.Tensor],
    ctc_loss_data_cls: Type[BaseCtcLossData],
) -> tf.Tensor:
    """Computes a version of CTC loss from logarothmic probabilities considered as independent parameters.

    Args:
        labels:             tf.Tensor, shape = [batch, max_label_length],       dtype = tf.int32
        logprobas:          tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
        label_length:       tf.Tensor, shape = [batch],                         dtype = tf.int32
        logit_length:       tf.Tensor, shape = [batch],                         dtype = tf.int32
        blank_index:        static integer >= 0
        ctc_loss_data_cls:  BaseCtcLossData class

    Returns:                tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
    """
    loss_data = ctc_loss_data_cls(
        labels=labels,
        logprobas=tf.stop_gradient(logprobas),
        label_length=label_length,
        logit_length=logit_length,
        blank_index=blank_index,
    )

    return loss_data.forward_fn(logprobas)


class BaseCtcLossData(ABC):
    """Base class for CTC loss data."""

    def __init__(
        self,
        labels: tf.Tensor,
        logprobas: tf.Tensor,
        label_length: tf.Tensor,
        logit_length: tf.Tensor,
        blank_index: Union[int, tf.Tensor],
        swap_memory: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._logprobas = logprobas
        self._original_label = labels
        self._logit_length = logit_length
        self._original_label_length = label_length
        self._verify_inputs()

        if isinstance(blank_index, (tf.Tensor, tf.Variable)):
            self._blank_index = blank_index
        else:
            self._blank_index = tf.constant(blank_index, dtype=tf.int32)

        self._swap_memory = swap_memory

    def _verify_inputs(self) -> None:
        assert len(self._logprobas.shape) == 3
        assert self._logprobas.dtype == tf.float32
        assert len(self._original_label.shape) == 2
        assert len(self._logit_length.shape) == 1
        assert len(self._original_label_length.shape) == 1

        assert self._logprobas.shape[0] == self._original_label.shape[0]
        assert self._logprobas.shape[0] == self._logit_length.shape[0]
        assert self._logprobas.shape[0] == self._original_label_length.shape[0]

    @tf.custom_gradient
    def forward_fn(self, unused_logprobas: tf.Tensor) -> tf.Tensor:
        """Forward pass of the loss function.

        Args:
            unused_logprobas: shape = [batch_size, max_logit_length, num_tokens]

        Returns: shape = [batch_size]
        """

        def backprop(d_loss):
            return expand_many_dims(d_loss, axes=[1, 2]) * self.gradient_fn(
                unused_logprobas
            )

        return self.loss, backprop

    @tf.custom_gradient
    def gradient_fn(self, unused_logprobas: tf.Tensor) -> tf.Tensor:
        """Gradient of loss w.r.t. input logits.

        Args:
            unused_logprobas: shape = [batch_size, max_logit_length, num_tokens]

        Returns: shape = [batch_size, max_logit_length, num_tokens]
        """

        def backprop(d_gradient):
            output = tf.reduce_sum(
                input_tensor=expand_many_dims(d_gradient, axes=[1, 2])
                * self._hessian_fn(unused_logprobas),
                axis=[3, 4],
            )
            return output

        return self.gradient, backprop

    @tf.custom_gradient
    def _hessian_fn(self, unused_logprobas: tf.Tensor) -> tf.Tensor:
        def backprop(d_hessian):
            raise NotImplementedError(
                "Third order derivative over the ctc loss function is not implemented."
            )

        return self.hessian, backprop

    @cached_property
    def hessian(self) -> tf.Tensor:
        """Calculates Hessian of loss w.r.t. input logits.

        Returns: tf.Tensor, shape = [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens]
        """
        alpha_gamma_term = self._combine_transition_probabilities(
            a=self.alpha[:, :-1], b=self.gamma[:, 1:]
        )
        # shape = [batch_size, max_logit_length, num_tokens, max_logit_length + 1, max_label_length + 1]
        alpha_gamma_beta_term = self._combine_transition_probabilities(
            a=alpha_gamma_term[:, :, :, :-1], b=self.beta[:, 1:]
        )
        # shape = [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens]
        alpha_gamma_beta_loss_term = (
            expand_many_dims(self.loss, axes=[1, 2, 3, 4]) + alpha_gamma_beta_term
        )
        # shape = [batch_size, max_logit_length, num_tokens]
        logit_length_x_num_tokens = self._max_logit_length * self._num_tokens
        first_term = tf.reshape(
            tf.linalg.set_diag(
                input=tf.reshape(
                    tensor=alpha_gamma_beta_loss_term,
                    shape=[
                        self._batch_size,
                        logit_length_x_num_tokens,
                        logit_length_x_num_tokens,
                    ],
                ),
                diagonal=tf.reshape(
                    tensor=self.logarithmic_logproba_gradient,
                    shape=[self._batch_size, logit_length_x_num_tokens],
                ),
            ),
            shape=tf.shape(alpha_gamma_beta_term),
        )

        mask = expand_many_dims(
            x=tf.linalg.band_part(
                tf.ones(shape=[self._max_logit_length] * 2, dtype=tf.bool), 0, -1
            ),
            axes=[0, 2, 4],
        )
        symmetrized_first_term = tf.where(
            condition=mask,
            x=first_term,
            y=tf.transpose(first_term, [0, 3, 4, 1, 2]),
        )
        # shape = [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens]
        hessian = -tf.exp(symmetrized_first_term) + expand_many_dims(
            self.gradient, [3, 4]
        ) * expand_many_dims(self.gradient, [1, 2])
        # shape = [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens]

        # Filter out samples with infinite loss
        hessian = tf.where(
            condition=expand_many_dims(self.loss == inf, [1, 2, 3, 4]),
            x=tf.zeros(shape=[1, 1, 1, 1, 1]),
            y=hessian,
        )
        # shape = [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens]

        # Filter out logits that beyond logits length
        hessian = tf.where(
            condition=expand_many_dims(self._logit_length_mask, axes=[2, 3, 4]),
            x=hessian,
            y=0.0,
        )
        hessian = tf.where(
            condition=expand_many_dims(self._logit_length_mask, axes=[1, 2, 4]),
            x=hessian,
            y=0.0,
        )

        return hessian

    @cached_property
    def gradient(self) -> tf.Tensor:
        """Gradient of loss w.r.t. input logits.

        Returns: shape = [batch_size, max_logit_length, num_tokens]
        """
        return -tf.exp(self.logarithmic_logproba_gradient)

    @cached_property
    def logarithmic_logproba_gradient(self) -> tf.Tensor:
        """Calculates logarithmic gradient of log loss w.r.t. input logarithmic probabilities.

        Returns: tf.Tensor, shape = [batch_size, max_logit_length, num_tokens]
        """
        logarithmic_logproba_gradient = tf.reshape(
            self.loss, [-1, 1, 1]
        ) + self._combine_transition_probabilities(
            a=self.alpha[:, :-1], b=self.beta[:, 1:]
        )
        # shape = [batch_size, max_logit_length, num_tokens]

        # Filter out samples infinite loss
        logarithmic_logproba_gradient = tf.where(
            condition=expand_many_dims(self.loss == inf, [1, 2]),
            x=-inf,
            y=logarithmic_logproba_gradient,
        )
        # shape = [batch_size, max_logit_length, num_tokens]

        # Filter out logits that beyond logits length
        logarithmic_logproba_gradient = apply_logarithmic_mask(
            tensor=logarithmic_logproba_gradient,
            mask=tf.expand_dims(self._logit_length_mask, axis=2),
        )
        # shape = [batch_size, max_logit_length, num_tokens]

        return logarithmic_logproba_gradient

    @property
    @abstractmethod
    def alpha(self) -> tf.Tensor:
        """Alpha tensor.

        Returns: shape = [batch_size, max_logit_length + 1, max_label_length + 1, ...]
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def beta(self) -> tf.Tensor:
        """Beta tensor.

        Returns: shape = [batch_size, max_logit_length + 1, max_label_length + 1, ...]
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def gamma(self) -> tf.Tensor:
        """Gamma tensor.

        Returns: shape =
            [batch_size, max_logit_length + 1, max_label_length + 1, max_logit_length + 1, max_label_length + 1, ...]
        """
        raise NotImplementedError()

    @cached_property
    def _expected_token_logproba(self) -> tf.Tensor:
        """Logarithmic probability to predict label token.

        Returns:shape = [batch_size, max_logit_length, max_label_length + 1]
        """
        label_logproba = tf.gather(
            params=self._logproba,
            indices=self._label,
            axis=2,
            batch_dims=1,
        )
        expected_token_logproba = apply_logarithmic_mask(
            label_logproba, tf.expand_dims(self._label_length_mask, axis=1)
        )
        # shape = [batch_size, max_logit_length, max_label_length + 1]
        return expected_token_logproba

    @property
    @abstractmethod
    def loss(self) -> tf.Tensor:
        """Samplewise loss function value that is minus logarithmic probability to predict label sequence.

        Returns:    tf.Tensor, shape = [batch_size]
        """
        raise NotImplementedError()

    @cached_property
    def _label_token_logproba(self) -> tf.Tensor:
        """shape = [batch_size, max_logit_length, max_label_length + 1]"""
        return tf.gather(
            params=self._logproba,
            indices=self._label,
            axis=2,
            batch_dims=1,
        )

    @cached_property
    def _blank_logproba(self):
        """Calculates logarithmic probability to predict blank token for given logit.

        Returns:    tf.Tensor, shape = [batch_size, max_logit_length]
        """
        return self._logproba[:, :, self._blank_token_index]

    @cached_property
    def _input_proba(self) -> tf.Tensor:
        """shape = [batch_size, input_logit_tensor_length, num_tokens], dtype = tf.float32"""
        return tf.exp(self._logproba)

    @cached_property
    def _logproba(self) -> tf.Tensor:
        mask = tf.expand_dims(
            tf.sequence_mask(lengths=self._logit_length, maxlen=self._max_logit_length),
            2,
        )
        blank_logprobas = tf.reshape(
            tf.math.log(tf.one_hot(self._blank_token_index, self._num_tokens)),
            shape=[1, 1, -1],
        )
        logprobas = tf.where(
            condition=mask,
            x=self._logprobas,
            y=blank_logprobas,
        )
        return logprobas

    @cached_property
    def _cleaned_label(self) -> tf.Tensor:
        """shape = [batch, max_label_length + 1]"""
        _ = self._max_label_length_plus_one
        labels = tf.cond(
            pred=tf.shape(self._original_label)[1] > self._max_label_length,
            true_fn=lambda: self._original_label[:, : self._max_label_length_plus_one],
            false_fn=lambda: pad_until(
                tensor=self._original_label,
                desired_size=self._max_label_length_plus_one,
                pad_value=self._pad_token_index,
                axis=1,
            ),
        )
        mask = tf.sequence_mask(
            lengths=self._original_label_length, maxlen=tf.shape(labels)[1]
        )
        blank_label = tf.ones_like(labels) * self._pad_token_index
        cleaned_label = tf.where(
            condition=mask,
            x=labels,
            y=blank_label,
        )
        return cleaned_label

    def _select_from_act(self, act: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """Takes tensor of acts act_{b, a, t, u, ...} and labels label_{b,u},
        where b is the batch index, t is the logit index, and u is the label index,
        and returns for each token index k the tensor

            output_{b,a,t,k,...} = logsumexp_u act_{b,a,t,u_k,...} * kroneker_delta(u_k = label_{b,u})

        that is logarithmic sum of exponents of acts for all u_k = label_{b,u}, given b, t and k.

        Args:
            act:    tf.Tensor, shape = [batch_size, dim_a, max_logit_length, max_label_length + 1, ...]
            label:  tf.Tensor, shape = [batch_size, max_label_length + 1]

        Returns:    tf.Tensor, shape = [batch_size, max_label_length + 1, num_tokens, ...]
        """
        data = smart_transpose(a=act, perm=[0, 3, 2, 1])
        # shape = [batch_size, max_label_length + 1, max_logit_length, dim_a, ...]
        data = tf.squeeze(
            input=smart_reshape(
                tensor=data,
                shape=[
                    1,
                    self._batch_size * self._max_label_length_plus_one,
                    self._max_logit_length,
                ],
            ),
            axis=0,
        )
        # shape = [batch_size * (max_label_length + 1), max_logit_length, dim_a, ...]

        segment_ids = tf.reshape(
            label + tf.expand_dims(tf.range(self._batch_size), 1) * self._num_tokens,
            shape=[-1],
        )
        # shape = [batch_size * (max_label_length + 1)]
        num_segments = self._batch_size * self._num_tokens

        output = unsorted_segment_logsumexp(
            data=data, segment_ids=segment_ids, num_segments=num_segments
        )
        # shape = [batch_size * num_tokens, max_logit_length, dim_a, ...]
        output = smart_reshape(
            tf.expand_dims(output, 0),
            [self._batch_size, self._num_tokens, self._max_logit_length],
        )
        # shape = [batch_size, num_tokens, max_logit_length, dim_a, ...]
        output = smart_transpose(output, [0, 3, 2, 1])
        # shape = [batch_size, dim_a, max_logit_length, num_tokens, ...]
        return output

    @cached_property
    def _max_logit_length_plus_one(self) -> tf.Tensor:
        return self._max_logit_length + tf.constant(1, dtype=tf.int32)

    @cached_property
    def _max_logit_length(self) -> tf.Tensor:
        return tf.shape(self._logprobas)[1]

    @cached_property
    def _max_label_length_plus_one(self) -> tf.Tensor:
        return self._max_label_length + tf.constant(1, dtype=tf.int32)

    @cached_property
    def _max_label_length(self) -> tf.Tensor:
        return reduce_max_with_default(
            self._original_label_length, default=tf.constant(0, dtype=tf.int32)
        )

    @cached_property
    def _pad_token_index(self) -> tf.Tensor:
        return self._blank_token_index

    @cached_property
    def _num_tokens(self) -> tf.Tensor:
        return tf.shape(self._logprobas)[2]

    @cached_property
    def _blank_token_index(self) -> tf.Tensor:
        return self._blank_index

    @cached_property
    def _logit_length_mask(self) -> tf.Tensor:
        """shape = [batch_size, max_logit_length]"""
        return tf.sequence_mask(
            lengths=self._logit_length,
            maxlen=self._max_logit_length,
        )

    @cached_property
    def _label_length_mask(self) -> tf.Tensor:
        """shape = [batch_size, max_label_length + 1], dtype = tf.bool"""
        return tf.sequence_mask(
            lengths=self._label_length, maxlen=self._max_label_length_plus_one
        )

    @property
    def _label_length(self) -> tf.Tensor:
        return self._original_label_length

    @cached_property
    def _preceded_label(self) -> tf.Tensor:
        """Preceded label. For example, for label "abc_" the sequence "_abc" is returned.

        Returns:    tf.Tensor, shape = [batch_size, max_label_length + 1]
        """
        return tf.roll(self._label, shift=1, axis=1)

    @cached_property
    def _label(self) -> tf.Tensor:
        """shape = [batch, max_label_length + 1]"""
        return self._cleaned_label

    @cached_property
    def _batch_size(self) -> tf.Tensor:
        return tf.shape(self._logprobas)[0]

    @abstractmethod
    def _combine_transition_probabilities(
        self, a: tf.Tensor, b: tf.Tensor
    ) -> tf.Tensor:
        """Given logarithmic probabilities a and b are merges like
        a, b -> log( exp a exp p exp b )
        """
        raise NotImplementedError()
