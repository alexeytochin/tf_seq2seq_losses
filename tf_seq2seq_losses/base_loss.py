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
from typing import Union, Type, Tuple, Callable
import tensorflow as tf
from cached_property import cached_property

from tf_seq2seq_losses.tools import logit_to_logproba, pad_until, reduce_max_with_default, subexp, \
    unsorted_segment_logsumexp, apply_logarithmic_mask


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
    @tf.custom_gradient
    def loss_fn(logits_arg: tf.Tensor) -> Tuple[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]:
        loss_data = ctc_loss_data_cls(
            labels=labels,
            logits=logits_arg,
            label_length=label_length,
            logit_length=logit_length,
            blank_index=blank_index,
        )
        return loss_data.log_loss, loss_data.backprop
    output = loss_fn(logits)
    return output


class BaseCtcLossData(ABC):
    """ Base class for CTC loss data. """
    def __init__(
            self,
            labels: tf.Tensor,
            logits: tf.Tensor,
            label_length: tf.Tensor,
            logit_length: tf.Tensor,
            blank_index: Union[int, tf.Tensor],
            swap_memory: bool = False,
    ):
        super().__init__()
        self._logits = logits
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
        assert len(self._logits.shape) == 3
        assert len(self._original_label.shape) == 2
        assert len(self._logit_length.shape) == 1
        assert len(self._original_label_length.shape) == 1

    def backprop(self, dloss: tf.Tensor) -> tf.Tensor:
        return tf.reshape(dloss, shape=[-1, 1, 1]) * self.gradient

    @cached_property
    def gradient(self) -> tf.Tensor:
        """Calculates gradient of log loss w.r.t. input logits.

        Returns: tf.Tensor, shape = [batch_size, max_logit_length, num_tokens]
        """
        sum_logarithmic_log_proba_gradient = \
            tf.reduce_logsumexp(self.logarithmic_log_proba_gradient, keepdims=True, axis=2)
        gradient = subexp(sum_logarithmic_log_proba_gradient + self.log_proba, self.logarithmic_log_proba_gradient)
        return gradient

    @cached_property
    def logarithmic_log_proba_gradient(self) -> tf.Tensor:
        """Calculates logarithmic gradient of log loss w.r.t. input logarithmic probabilities.

        Returns: tf.Tensor, shape = [batch_size, max_logit_length, num_tokens]
        """
        blank_mask = tf.reshape(self.blank_token_index == tf.range(self.num_tokens), [1, 1, -1])
        # shape = [1, 1, num_tokens]
        logarithmic_log_proba_gradient = \
            tf.reshape(self.log_loss, [-1, 1, 1]) \
            + tf.where(
                condition=blank_mask,
                x=tf.expand_dims(self.blank_grad_term, axis=2),
                y=self.non_blank_grad_term,
            )
        # shape = [batch_size, max_logit_length, num_tokens]

        logit_length_mask = self.label_length <= self._logit_length
        logarithmic_log_proba_gradient = tf.where(
            condition=tf.reshape(logit_length_mask, [-1, 1, 1]),
            x=logarithmic_log_proba_gradient,
            y=tf.math.log(tf.zeros_like(logarithmic_log_proba_gradient))
        )
        # shape = [batch_size, max_logit_length, num_tokens]

        return logarithmic_log_proba_gradient

    @property
    @abstractmethod
    def non_blank_grad_term(self) -> tf.Tensor:
        """Calculates gradient term log [sum exp(alpha) * exp(log_proba) * exp(beta)]
        for non-blank tokens.
        The values at output[:, :, self._token_index] are to be ignored.

        Returns: tf.Tensor, shape = [batch_size, max_logit_length, num_tokens]
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def blank_grad_term(self) -> tf.Tensor:
        """Calculates gradient term log [sum exp(alpha) * exp(log_proba) * exp(beta)] for the blank token.

        Returns: tf.Tensor, shape = [batch_size, max_logit_length]
        """
        raise NotImplementedError()

    @cached_property
    def expected_token_log_proba(self) -> tf.Tensor:
        """Logarithmic probability to predict label token.

        Returns:shape = [batch_size, max_logit_length, max_label_length + 1]
        """
        label_log_proba = tf.gather(
            params=self.log_proba,
            indices=self.label,
            axis=2,
            batch_dims=1,
        )
        expected_token_log_proba = \
            apply_logarithmic_mask(label_log_proba, tf.expand_dims(self.label_length_mask, axis=1))
        # shape = [batch_size, max_logit_length, max_label_length + 1]
        return expected_token_log_proba

    @property
    @abstractmethod
    def log_loss(self) -> tf.Tensor:
        """ shape = [batch_size] """
        raise NotImplementedError()

    @cached_property
    def label_token_log_proba(self) -> tf.Tensor:
        """ shape = [batch_size, max_logit_length, max_label_length + 1] """
        return tf.gather(
            params=self.log_proba,
            indices=self.label,
            axis=2,
            batch_dims=1,
        )

    @cached_property
    def blank_log_proba(self):
        """Calculates logarithmic probability to predict blank token for given logit.

        Returns:    tf.Tensor, shape = [batch_size, max_logit_length]
        """
        return self.log_proba[:, :, self.blank_token_index]

    @cached_property
    def input_proba(self) -> tf.Tensor:
        """ shape = [batch_size, input_logit_tensor_length, num_tokens], dtype = tf.float32 """
        return tf.exp(self.log_proba)

    @cached_property
    def log_proba(self) -> tf.Tensor:
        """ shape = [batch_size, input_logit_tensor_length, num_tokens], dtype = tf.float32 """
        mask = tf.expand_dims(tf.sequence_mask(lengths=self._logit_length, maxlen=self.max_logit_length), 2)
        blank_logit = tf.reshape(tf.math.log(tf.one_hot(self.blank_token_index, self.num_tokens)), shape=[1, 1, -1])
        logits_clean = tf.where(
            condition=mask,
            x=self.logit,
            y=blank_logit,
        )
        log_probas = logit_to_logproba(logit=logits_clean, axis=2)
        return log_probas

    @cached_property
    def cleaned_label(self) -> tf.Tensor:
        """ shape = [batch, max_label_length + 1] """
        labels = tf.cond(
            pred=tf.shape(self._original_label)[1] > self.max_label_length,
            true_fn=lambda: self._original_label[:, :self.max_label_length + 1],
            false_fn=lambda: pad_until(
                tensor=self._original_label,
                desired_size=self.max_label_length + 1,
                pad_value=self.pad_token_index,
                axis=1
            )
        )
        mask = tf.sequence_mask(lengths=self._original_label_length, maxlen=tf.shape(labels)[1])
        blank_label = tf.ones_like(labels) * self.pad_token_index
        cleaned_label = tf.where(
            condition=mask,
            x=labels,
            y=blank_label,
        )
        return cleaned_label

    def select_from_act(self, act: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """Takes tensor of acts act_{b,t,u} and labels label_{b,u},
        where b is the batch index, t is the logit index, and u is the label index,
        and returns for each token index k the tensor

            output_{b, t, k} = logsumexp_u act_{b, t, u_k} * delta(u_k = label_{b,u})

        that is logarithmic sum of exponents of acts for all u_k = label_{b,u}, given b, t, k.

        Args:
            act:    tf.Tensor, shape = [batch_size, max_logit_length, max_label_length + 1]
            label:  tf.Tensor, shape = [batch_size, max_label_length + 1]

        Returns:    tf.Tensor, shape = [batch_size, max_label_length + 1, num_tokens]
        """
        data = tf.transpose(act, [0, 2, 1])
        data = tf.reshape(data, shape=[self.batch_size * (self.max_label_length + 1), self.max_logit_length])
        # shape = [batch_size * (max_label_length + 1), max_logit_length]
        segment_ids = tf.reshape(label + tf.expand_dims(tf.range(self.batch_size), 1) * self.num_tokens, shape=[-1])
        # shape = [batch_size * (max_label_length + 1)]
        num_segments = self.batch_size * self.num_tokens

        output = unsorted_segment_logsumexp(data=data, segment_ids=segment_ids, num_segments=num_segments)
        # shape = [batch_size * num_tokens, max_logit_length]

        output = tf.reshape(output, [self.batch_size, self.num_tokens, self.max_logit_length])
        output = tf.transpose(output, [0, 2, 1])
        return output

    @cached_property
    def max_logit_length(self) -> tf.Tensor:
        return tf.shape(self._logits)[1]

    @cached_property
    def max_label_length(self) -> tf.Tensor:
        return reduce_max_with_default(self._original_label_length, default=tf.constant(0, dtype=tf.int32))

    @cached_property
    def pad_token_index(self) -> tf.Tensor:
        return self.blank_token_index

    @cached_property
    def num_tokens(self) -> tf.Tensor:
        return tf.shape(self._logits)[2]

    @cached_property
    def blank_token_index(self) -> tf.Tensor:
        return self._blank_index

    @property
    def logit(self):
        """ shape = [batch_size, input_logit_tensor_length, num_tokens] """
        return self._logits

    @cached_property
    def label_length_mask(self) -> tf.Tensor:
        """ shape = [batch_size, max_label_length + 1], dtype = tf.bool """
        return tf.sequence_mask(lengths=self.label_length, maxlen=self.max_label_length + 1)

    @property
    def label_length(self) -> tf.Tensor:
        return self._original_label_length

    @cached_property
    def preceded_label(self) -> tf.Tensor:
        """Preceded label. For example, for label "abc_" the sequence "_abc" is returned.

        Returns:    tf.Tensor, shape = [batch_size, max_label_length + 1]
        """
        return tf.roll(self.label, shift=1, axis=1)

    @cached_property
    def label(self) -> tf.Tensor:
        """ shape = [batch, max_label_length + 1] """
        return self.cleaned_label

    @cached_property
    def batch_size(self) -> tf.Tensor:
        return tf.shape(self._logits)[0]
