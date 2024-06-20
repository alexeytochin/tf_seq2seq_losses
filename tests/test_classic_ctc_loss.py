"""Tests for the classic CTC loss."""

# ==============================================================================
# Copyright 2022 Alexey Tochin
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
import numpy as np
import tensorflow as tf

from tf_seq2seq_losses.base_loss import ctc_loss_from_logproba
from tf_seq2seq_losses.classic_ctc_loss import ClassicCtcLossData, classic_ctc_loss
from tf_seq2seq_losses.tools import logit_to_logproba

from tests.common import generate_ctc_loss_inputs
from tests.test_ctc_losses import TestCtcLoss
from tests.finite_difference import finite_difference_batch_jacobian


class TestClassicCtcLoss(TestCtcLoss):
    """Tests for the classic CTC loss."""

    def test_single_logit_case(self):
        """Test for the case of a single logit."""
        logits = tf.math.log(tf.constant([[[0, 1, 0]]], dtype=tf.float32))
        labels = tf.constant([[1]], dtype=tf.int32)
        length_label = tf.constant([1], dtype=tf.int32)
        length_logit = tf.constant([1], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)
        logproba = logit_to_logproba(logit=logits, axis=2)

        loss_data = ClassicCtcLossData(
            logprobas=logproba,
            labels=labels,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assert_tensors_almost_equal(
            tf.constant([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=tf.float32),
            tf.exp(loss_data.alpha),
            places=None,
        )
        self.assert_tensors_almost_equal(
            tf.constant([[[1, 1], [0, 1]], [[0, 0], [1, 1]]], dtype=tf.float32),
            tf.exp(loss_data.beta),
            places=None,
        )
        self.assertEqual(0.0, loss_data.loss.numpy()[0])
        self.assert_tensors_almost_equal(
            first=tf.constant([[[0.0, 1.0, 0.0]]]),
            second=tf.exp(loss_data.logarithmic_logproba_gradient),
            places=6,
        )

    def test_closed_state(self):
        """Test for the case of a closed state."""
        logit = tf.math.log(tf.constant([[[0, 1, 0], [1, 0, 0]]], dtype=tf.float32))
        label = tf.constant([[1]], dtype=tf.int32)
        length_label = tf.constant([1], dtype=tf.int32)
        length_logit = tf.constant([2], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)
        logproba = logit_to_logproba(logit=logit, axis=2)

        loss_data = ClassicCtcLossData(
            logprobas=logproba,
            labels=label,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assert_tensors_almost_equal(
            tf.constant(
                [[[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[0, 0], [1, 0]]]],
                dtype=tf.float32,
            ),
            tf.exp(loss_data.alpha),
            places=None,
        )
        self.assert_tensors_almost_equal(
            tf.constant(
                [[[[1, 1], [0, 1]], [[0, 0], [1, 1]], [[0, 0], [1, 1]]]],
                dtype=tf.float32,
            ),
            tf.exp(loss_data.beta),
            places=None,
        )
        self.assertEqual(0.0, loss_data.loss.numpy()[0])
        self.assert_tensors_almost_equal(
            first=tf.constant([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            second=tf.exp(loss_data.logarithmic_logproba_gradient),
            places=6,
        )

    def test_classic_loss_simple_case(self):
        """Test for the simple case of the classic CTC loss."""
        logit = tf.math.log(
            tf.constant(
                [[[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]]],
                dtype=tf.float32,
            )
        )
        label = tf.constant([[1, 2, 2, 1]], dtype=tf.int32)
        length_label = tf.constant([4], dtype=tf.int32)
        length_logit = tf.constant([5], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)
        logproba = logit_to_logproba(logit=logit, axis=2)

        loss_data = ClassicCtcLossData(
            logprobas=logproba,
            labels=label,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assertLess(loss_data.loss.numpy()[0].item(), 1e-6)
        self.assert_tensors_almost_equal(
            first=tf.constant(
                [
                    [
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                    ]
                ]
            ),
            second=tf.exp(loss_data.logarithmic_logproba_gradient),
            places=6,
        )

    def test_alpha_beta_sum(self):
        """Test for the sum of alpha and beta."""
        input_dict = generate_ctc_loss_inputs(
            max_logit_length=6, batch_size=1, random_seed=0, num_tokens=5, blank_index=0
        )
        loss_data = ClassicCtcLossData(
            labels=input_dict["labels"],
            logprobas=input_dict["logprobas"],
            label_length=input_dict["label_length"],
            logit_length=input_dict["logit_length"],
            blank_index=0,
        )

        # Sums along U of products alpha * beta that is supposed to be equal to the loss function
        sums = tf.reduce_logsumexp(loss_data.alpha + loss_data.beta, axis=[2, 3])

        # We verify that the values of the sums `a` equal to the loss up to a sign.
        self.assert_tensors_almost_equal(
            first=-tf.expand_dims(loss_data.loss, 1),
            second=sums,
            places=6,
        )

    def test_length_two_case(self):
        """Test for the case of a length two label."""
        batch_size = 2
        num_tokens = 3
        blank_index = 0
        logits = tf.zeros(shape=[batch_size, 2, num_tokens])
        labels = tf.constant([[1, 2], [1, 2]])
        label_length = tf.constant([2, 1], dtype=tf.int32)
        logit_length = tf.constant([2, 2], dtype=tf.int32)
        logproba = logit_to_logproba(logit=logits, axis=2)

        loss_data = ClassicCtcLossData(
            logprobas=logproba,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank_index=blank_index,
        )

        self.assertAlmostEqual(-np.log(1 / 3 * 1 / 3), loss_data.loss.numpy()[0], 6)
        self.assertAlmostEqual(-np.log(3 * 1 / 3 * 1 / 3), loss_data.loss.numpy()[1], 6)
        self.assert_tensors_almost_equal(
            first=tf.constant(
                [
                    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                    [[-1 / 3, -2 / 3, 0.0], [-1 / 3, -2 / 3, 0.0]],
                ]
            ),
            second=loss_data.gradient,
            places=6,
        )

    def test_too_short_logit(self):
        """Test for the case of a too short logit."""
        batch_size = 1
        num_tokens = 3
        blank_index = 0
        max_logit_length = 2
        logits = tf.zeros(shape=[batch_size, max_logit_length, num_tokens])
        labels = tf.constant([[1, 1]])
        label_length = tf.constant([2], dtype=tf.int32)
        logit_length = tf.constant([max_logit_length], dtype=tf.int32)
        logproba = logit_to_logproba(logit=logits, axis=2)

        loss_data = ClassicCtcLossData(
            logprobas=logproba,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank_index=blank_index,
        )

        self.assert_tensors_almost_equal(
            tf.constant([np.inf]), loss_data.loss, places=None
        )
        self.assert_tensors_almost_equal(
            tf.zeros(shape=[batch_size, max_logit_length, num_tokens]),
            loss_data.gradient,
            places=None,
        )
        self.assert_tensors_almost_equal(
            tf.zeros(
                shape=[
                    batch_size,
                    max_logit_length,
                    num_tokens,
                    max_logit_length,
                    num_tokens,
                ]
            ),
            loss_data.hessian,
            places=None,
        )

    def test_repeated_token(self):
        """Test for the case of a repeated token."""
        num_tokens = 3
        blank_index = 0
        logits = tf.zeros(shape=[1, 3, num_tokens])
        labels = tf.constant([[1, 1]])
        label_length = tf.constant([2], dtype=tf.int32)
        logit_length = tf.constant([3], dtype=tf.int32)
        logproba = logit_to_logproba(logit=logits, axis=2)

        loss_data = ClassicCtcLossData(
            logprobas=logproba,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank_index=blank_index,
        )

        # Label "aa" corresponds to a single paths: "a_a" with probability 3 ** -3
        self.assertAlmostEqual(np.log(3**3), loss_data.loss.numpy()[0], 6)

    def test_single_token(self):
        """Test for the case of a single token."""
        num_tokens = 3
        blank_index = 0
        logits = tf.zeros(shape=[1, 3, num_tokens])
        labels = tf.constant([[1]])
        label_length = tf.constant([1], dtype=tf.int32)
        logit_length = tf.constant([3], dtype=tf.int32)
        logproba = logit_to_logproba(logit=logits, axis=2)

        loss_data = ClassicCtcLossData(
            logprobas=logproba,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank_index=blank_index,
        )

        # Label "a" corresponds to 6 paths: "a__", "_a_", "__a", "aa_", "_aa", and "aaa" with equal probability 3 ** -3
        self.assertAlmostEqual(np.log(3**3 / 6), loss_data.loss.numpy()[0], 6)

    def test_wrong_prediction_case(self):
        """Test for the case of a wrong prediction."""
        logit = tf.constant([[[0, 0, 1]]], dtype=tf.float32) * 100
        label = tf.constant([[1]], dtype=tf.int32)
        length_label = tf.constant([1], dtype=tf.int32)
        length_logit = tf.constant([1], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)
        logproba = logit_to_logproba(logit=logit, axis=2)

        loss_data = ClassicCtcLossData(
            logprobas=logproba,
            labels=label,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assert_tensors_almost_equal(
            tf.constant([[[0.0, -1.0, 0.0]]], dtype=tf.float32),
            loss_data.gradient,
            places=None,
        )
        self.assertEqual(100.0, loss_data.loss.numpy()[0])

    def test_zero_batch_size_with_autograph(self):
        """Test for the case of a zero batch size with autograph."""
        logits = tf.zeros(shape=[0, 4, 3], dtype=tf.float32)
        labels = tf.zeros(shape=[0, 2], dtype=tf.int32)
        label_length = tf.zeros(shape=[0], dtype=tf.int32)
        logit_length = tf.zeros(shape=[0], dtype=tf.int32)

        @tf.function
        def func():
            with tf.GradientTape() as tape:
                tape.watch([logits])
                loss_samplewise_ = classic_ctc_loss(
                    labels, logits, label_length, logit_length, 0
                )
                loss = tf.reduce_sum(loss_samplewise_)
            gradient_ = tape.gradient(loss, sources=logits)
            return loss_samplewise_, gradient_

        loss_samplewise, gradient = func()

        self.assertEqual([0], loss_samplewise.shape)
        self.assertEqual([0, 4, 3], gradient.shape)

    def test_compare_forward_with_tf_implementation(self):
        """Test for the comparison of the forward pass with the TensorFlow implementation."""
        input_dict = generate_ctc_loss_inputs(
            max_logit_length=20,
            batch_size=8,
            random_seed=0,
            num_tokens=8,
            blank_index=0,
        )
        tf_ctc_loss = tf.nn.ctc_loss(
            labels=input_dict["labels"],
            logits=input_dict["logits"],
            label_length=input_dict["label_length"],
            logit_length=input_dict["logit_length"],
            logits_time_major=False,
            blank_index=0,
        )

        local_ctc_loss = classic_ctc_loss(
            labels=input_dict["labels"],
            logits=input_dict["logits"],
            label_length=input_dict["label_length"],
            logit_length=input_dict["logit_length"],
            blank_index=0,
        )

        self.assert_tensors_almost_equal(tf_ctc_loss, local_ctc_loss, 5)

    def test_compare_gradient_with_tf_implementation(self):
        """Test for the comparison of the gradient with auto gradient."""
        blank_index = 0
        input_dict = generate_ctc_loss_inputs(
            max_logit_length=64,
            batch_size=8,
            random_seed=0,
            num_tokens=10,
            blank_index=blank_index,
        )

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([input_dict["logits"]])
            tf_loss = tf.nn.ctc_loss(
                labels=input_dict["labels"],
                logits=input_dict["logits"],
                label_length=input_dict["label_length"],
                logit_length=input_dict["logit_length"],
                logits_time_major=False,
                blank_index=blank_index,
            )
            testing_loss = classic_ctc_loss(
                labels=input_dict["labels"],
                logits=input_dict["logits"],
                label_length=input_dict["label_length"],
                logit_length=input_dict["logit_length"],
                blank_index=blank_index,
            )
        tf_version_gradient = tape.gradient(tf_loss, input_dict["logits"])
        classic_version_gradient = tape.gradient(testing_loss, input_dict["logits"])

        self.assert_tensors_almost_equal(
            tf_version_gradient, classic_version_gradient, 4
        )

    def test_gradient_vs_finite_difference(self):
        """Test for the comparison of the gradient with the finite difference."""
        blank_index = 0
        input_dict = generate_ctc_loss_inputs(
            max_logit_length=16,
            batch_size=1,
            random_seed=0,
            num_tokens=4,
            blank_index=blank_index,
        )
        logits = input_dict["logits"]

        def loss_fn(logits_):
            return classic_ctc_loss(
                labels=input_dict["labels"],
                logits=logits_,
                label_length=input_dict["label_length"],
                logit_length=input_dict["logit_length"],
                blank_index=blank_index,
            )

        gradient_numerical = finite_difference_batch_jacobian(
            func=loss_fn, x=logits, epsilon=1e-3
        )

        with tf.GradientTape() as tape:
            tape.watch([logits])
            loss = tf.reduce_sum(loss_fn(logits))
        gradient_analytic = tape.gradient(loss, sources=logits)

        self.assert_tensors_almost_equal(gradient_numerical, gradient_analytic, 2)

    def test_readme_example(self):
        """Test for the example from the README."""
        batch_size = 1
        num_token = 3
        logit_length = 5

        _ = classic_ctc_loss(
            labels=tf.constant([[1, 2, 2, 1]], dtype=tf.int32),
            logits=tf.zeros(
                shape=[batch_size, logit_length, num_token], dtype=tf.float32
            ),
            label_length=tf.constant([4], dtype=tf.int32),
            logit_length=tf.constant([logit_length], dtype=tf.int32),
            blank_index=0,
        )

    def test_second_derivative_shape(self):
        """Test for the shape of the second derivative."""
        batch_size = 2
        num_tokens = 3
        max_logit_length = 4
        input_dict = generate_ctc_loss_inputs(
            max_logit_length=max_logit_length,
            batch_size=batch_size,
            random_seed=0,
            num_tokens=num_tokens,
            blank_index=0,
        )
        logprobas = input_dict["logprobas"]

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([logprobas])
            with tf.GradientTape() as tape2:
                tape2.watch([logprobas])
                loss = ctc_loss_from_logproba(
                    labels=input_dict["labels"],
                    logprobas=logprobas,
                    label_length=input_dict["label_length"],
                    logit_length=input_dict["logit_length"],
                    blank_index=0,
                    ctc_loss_data_cls=ClassicCtcLossData,
                )
            gradient_analytic = tape2.gradient(loss, sources=logprobas)
        hessian_analytic = tape1.batch_jacobian(
            gradient_analytic, source=logprobas, experimental_use_pfor=False
        )

        self.assertEqual(
            [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens],
            list(hessian_analytic.shape),
        )

    def test_hessian_vs_finite_difference(self):
        """Test for the comparison of the Hessian with the finite difference."""
        input_dict = generate_ctc_loss_inputs(
            max_logit_length=4, batch_size=2, random_seed=0, num_tokens=2, blank_index=0
        )
        logits = input_dict["logits"]

        def gradient_fn(logits_):
            with tf.GradientTape() as tape_:
                tape_.watch([logits_])
                loss = tf.reduce_sum(
                    classic_ctc_loss(
                        labels=input_dict["labels"],
                        logits=logits_,
                        label_length=input_dict["label_length"],
                        logit_length=input_dict["logit_length"],
                        blank_index=0,
                    )
                )
            gradient_ = tape_.gradient(loss, sources=logits_)
            # shape = [batch_size, logit_length, num_tokens]
            return gradient_

        hessian_numerical = finite_difference_batch_jacobian(
            func=gradient_fn, x=logits, epsilon=1e-4
        )
        # shape = [batch_size, logit_length, num_tokens, logit_length, num_tokens]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([logits])
            gradient = gradient_fn(logits)
        hessain_analytic = tape.batch_jacobian(
            gradient, source=logits, experimental_use_pfor=False
        )

        self.assert_tensors_almost_equal(hessian_numerical, hessain_analytic, 2)
