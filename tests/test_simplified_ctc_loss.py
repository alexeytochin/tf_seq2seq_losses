"""Tests for simplified CTC loss function."""

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
import unittest

import numpy as np
import tensorflow as tf

from tests.common import generate_ctc_loss_inputs
from tests.test_ctc_losses import TestCtcLoss
from tests.finite_difference import finite_difference_batch_jacobian
from tf_seq2seq_losses.simplified_ctc_loss import (
    simplified_ctc_loss,
    SimplifiedCtcLossData,
)
from tf_seq2seq_losses.tools import logit_to_logproba


class TestSimplifiedCtcLoss(TestCtcLoss):
    """Tests for the simplified CTC loss."""

    def test_simple_case(self):
        """Test for a simple case."""
        logits = tf.math.log(
            tf.constant(
                [[[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]],
                dtype=tf.float32,
            )
        )
        labels = tf.constant([[1, 2, 1]], dtype=tf.int32)
        length_label = tf.constant([3], dtype=tf.int32)
        length_logit = tf.constant([5], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)
        logproba = logit_to_logproba(logit=logits, axis=2)

        loss_session = SimplifiedCtcLossData(
            logprobas=logproba,
            labels=labels,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assertTrue(
            tf.reduce_all(
                tf.exp(loss_session.alpha)
                == tf.constant(
                    [
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ]
                )
            ).numpy()
        )
        self.assertTrue(
            tf.reduce_all(
                tf.exp(loss_session.beta)
                == tf.constant(
                    [
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ]
                )
            ).numpy()
        )
        self.assertLess(loss_session.loss.numpy()[0].item(), 1e-6)

    def test_non_zero_blank_index(self):
        """Test for a non-zero blank index."""
        logits = tf.math.log(
            tf.constant(
                [[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]],
                dtype=tf.float32,
            )
        )
        labels = tf.constant([[0, 2, 0]], dtype=tf.int32)
        length_label = tf.constant([3], dtype=tf.int32)
        length_logit = tf.constant([5], dtype=tf.int32)
        blank_index = tf.constant(1, dtype=tf.int32)
        logproba = logit_to_logproba(logit=logits, axis=2)

        loss_session = SimplifiedCtcLossData(
            logprobas=logproba,
            labels=labels,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assertLess(loss_session.loss.numpy()[0].item(), 1e-6)

    def test_shorter_logit_and_label_length(self):
        """Test for shorter logit and label length."""
        logits = tf.math.log(
            tf.constant(
                [[[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]], dtype=tf.float32
            )
        )
        labels = tf.constant([[1, 0]], dtype=tf.int32)
        logit_length = tf.constant([3], dtype=tf.int32)
        label_length = tf.constant([1], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)
        logproba = logit_to_logproba(logit=logits, axis=2)

        loss_session = SimplifiedCtcLossData(
            logprobas=logproba,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank_index=blank_index,
        )

        self.assertEqual([0], loss_session.loss.numpy().tolist())

    def test_label_length_bigger_then_logit_length(self):
        """Test for label length bigger than logit length."""
        logits = tf.constant([[[0, 0, 0]]], dtype=tf.float32)
        labels = tf.constant([[1, 2]], dtype=tf.int32)
        logit_length = tf.constant([1], dtype=tf.int32)
        label_length = tf.constant([2], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)
        logproba = logit_to_logproba(logit=logits, axis=2)

        loss_session = SimplifiedCtcLossData(
            logprobas=logproba,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank_index=blank_index,
        )

        self.assertEqual(np.inf, loss_session.loss.numpy()[0])
        self.assertEqual(
            np.zeros(shape=[1, 1, 3]).tolist(), loss_session.gradient.numpy().tolist()
        )

    def test_large_loss(self):
        """Test for a large loss."""
        logits = tf.constant([[[1e10, 0.0, 0.0]]], dtype=tf.float32)
        labels = tf.constant([[1]], dtype=tf.int32)
        length_label = tf.constant([1], dtype=tf.int32)
        length_logit = tf.constant([1], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)
        logproba = logit_to_logproba(logit=logits, axis=2)

        loss_session = SimplifiedCtcLossData(
            logprobas=logproba,
            labels=labels,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assertEqual(1e10, loss_session.loss.numpy()[0])
        self.assertEqual([[[0.0, -1.0, 0.0]]], loss_session.gradient.numpy().tolist())
        self.assert_tensors_almost_equal(
            tf.constant([[[0.0, -1.0, 0.0]]]), loss_session.gradient, places=6
        )

    def test_alpha_beta_sum(self):
        """Test for the sum of alpha and beta."""
        input_dict = generate_ctc_loss_inputs(
            max_logit_length=6, batch_size=1, random_seed=1, num_tokens=5, blank_index=0
        )
        loss_session = SimplifiedCtcLossData(
            logprobas=input_dict["logprobas"],
            labels=input_dict["labels"],
            logit_length=input_dict["logit_length"],
            label_length=input_dict["label_length"],
            blank_index=0,
        )

        # Sums along U of products alpha * beta that is supposed to be equal to the loss function
        sums = tf.reduce_logsumexp(loss_session.alpha + loss_session.beta, axis=2)

        # We verify that the values of the sums an equal to the loss up to a sign.
        self.assert_tensors_almost_equal(
            first=-tf.expand_dims(loss_session.loss, 1),
            second=sums,
            places=6,
        )

    def test_length_one(self):
        """Test for label and logit the lengths of one."""
        batch_size = 1
        num_tokens = 3
        blank_index = 0
        logits = tf.zeros(shape=[batch_size, 1, num_tokens])
        labels = tf.constant([[1]])
        label_length = tf.constant([1], dtype=tf.int32)
        logit_length = tf.constant([1], dtype=tf.int32)
        logprobas = logit_to_logproba(logit=logits, axis=2)

        loss_data = SimplifiedCtcLossData(
            logprobas=logprobas,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank_index=blank_index,
        )

        self.assert_tensors_almost_equal(np.log(num_tokens), loss_data.loss[0], 8)
        self.assert_tensors_almost_equal(
            first=tf.constant([[[0.0, -1.0, 0.0]]]), second=loss_data.gradient, places=6
        )

    def test_length_two(self):
        """Test for label and logit the lengths of two."""
        batch_size = 1
        num_tokens = 3
        blank_index = 0
        logits = tf.zeros(shape=[batch_size, 2, num_tokens])
        labels = tf.constant([[1, 2]])
        label_length = tf.constant([2], dtype=tf.int32)
        logit_length = tf.constant([2], dtype=tf.int32)
        logprobas = logit_to_logproba(logit=logits, axis=2)

        loss_session = SimplifiedCtcLossData(
            logprobas=logprobas,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank_index=blank_index,
        )

        self.assert_tensors_almost_equal(
            2 * np.log(num_tokens), loss_session.loss[0], 8
        )
        self.assert_tensors_almost_equal(
            first=tf.constant([[[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]]),
            second=loss_session.gradient,
            places=6,
        )

    @unittest.skip("fix_finite_difference")
    def test_gradient_with_finite_difference(self):
        """Test for the gradient with finite difference."""
        blank_index = 0
        input_dict = generate_ctc_loss_inputs(
            max_logit_length=4,
            batch_size=1,
            random_seed=0,
            num_tokens=3,
            blank_index=blank_index,
        )
        logits = input_dict["logits"]

        def loss_fn(logits_):
            return tf.reduce_sum(
                simplified_ctc_loss(
                    labels=input_dict["labels"],
                    logits=logits_,
                    label_length=input_dict["label_length"],
                    logit_length=input_dict["logit_length"],
                    blank_index=blank_index,
                )
            )

        gradient_numerical = finite_difference_batch_jacobian(
            func=loss_fn, x=logits, epsilon=1e-5
        )

        with tf.GradientTape() as tape:
            tape.watch([logits])
            loss = tf.reduce_sum(loss_fn(logits))
        gradient_analytic = tape.gradient(loss, sources=logits)

        self.assert_tensors_almost_equal(gradient_numerical, gradient_analytic, 1)

    def test_autograph(self):
        """Test that TensorFlow graph can be built form simplified_ctc_loss and its gradient function."""

        @tf.function
        def func() -> tf.Tensor:
            input_dict = generate_ctc_loss_inputs(
                max_logit_length=6,
                batch_size=2,
                random_seed=0,
                num_tokens=3,
                blank_index=0,
            )

            with tf.GradientTape() as tape:
                tape.watch([input_dict["logits"]])
                output = simplified_ctc_loss(
                    labels=input_dict["labels"],
                    logits=input_dict["logits"],
                    label_length=input_dict["label_length"],
                    logit_length=input_dict["logit_length"],
                    blank_index=0,
                )
                loss = tf.reduce_mean(output)
            gradient = tape.gradient(loss, sources=input_dict["logits"])
            return gradient

        # This should not raise an exception
        func()

    def test_zero_logit_length(self):
        """Test for zero logit length."""
        logits = tf.zeros(shape=[1, 0, 3])
        labels = tf.constant([[1, 2]])
        label_length = tf.constant([2], dtype=tf.int32)
        logit_length = tf.constant([2], dtype=tf.int32)

        @tf.function
        def func():
            with tf.GradientTape() as tape:
                tape.watch([logits])
                output = simplified_ctc_loss(
                    labels, logits, label_length, logit_length, 0
                )
                loss = tf.reduce_mean(output)
            gradient = tape.gradient(loss, sources=logits)
            return loss, gradient

        loss, gradient = func()

        self.assertEqual(np.inf, loss.numpy())
        self.assertEqual([1, 0, 3], list(gradient.shape))

    def test_zero_batch_size(self):
        """Test for zero batch size."""
        logits = tf.zeros(shape=[0, 4, 3], dtype=tf.float32)
        labels = tf.zeros(shape=[0, 2], dtype=tf.int32)
        label_length = tf.zeros(shape=[0], dtype=tf.int32)
        logit_length = tf.zeros(shape=[0], dtype=tf.int32)

        @tf.function
        def func():
            with tf.GradientTape() as tape:
                tape.watch([logits])
                loss_samplewise = simplified_ctc_loss(
                    labels, logits, label_length, logit_length, 0
                )
                loss = tf.reduce_sum(loss_samplewise)
            gradient = tape.gradient(loss, sources=logits)
            return loss_samplewise, gradient

        loss_samplewise, gradient = func()

        self.assertEqual([0], loss_samplewise.shape)
        self.assertEqual([0, 4, 3], gradient.shape)
