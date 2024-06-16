"""Tests for CTC losses."""

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

import tensorflow as tf

from tests.common import generate_ctc_loss_inputs
from tests.test_ctc_losses import TestCtcLoss
from tests.finite_difference import finite_difference_batch_jacobian
from tf_seq2seq_losses import classic_ctc_loss
from tf_seq2seq_losses.base_loss import ctc_loss_from_logproba
from tf_seq2seq_losses.simplified_ctc_loss import (
    SimplifiedCtcLossData,
    simplified_ctc_loss,
)
from tf_seq2seq_losses.tools import logit_to_logproba


class TestSimplifiedCtcLoss(TestCtcLoss):
    """Tests for the simplified CTC loss."""

    def test_single_logit_case(self):
        """Test for the case with a single logit."""
        logits = tf.math.log(tf.constant([[[1 / 3, 1 / 3, 1 / 3]]], dtype=tf.float32))
        labels = tf.constant([[1]], dtype=tf.int32)
        length_label = tf.constant([1], dtype=tf.int32)
        length_logit = tf.constant([1], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)
        logproba = logit_to_logproba(logit=logits, axis=2)

        loss_data = SimplifiedCtcLossData(
            logprobas=logproba,
            labels=labels,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assert_tensors_almost_equal(
            first=tf.constant([[[0.0, -1.0, 0.0]]]), second=loss_data.gradient, places=6
        )

        self.assert_tensors_almost_equal(
            tf.zeros(shape=[1, 1, 3, 1, 3]), loss_data.hessian, places=6
        )

    def test_simple_case(self):
        """Test for the simple case."""
        logit = tf.math.log(
            tf.constant(
                [[[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]],
                dtype=tf.float32,
            )
        )
        label = tf.constant([[1, 2, 1]], dtype=tf.int32)
        length_label = tf.constant([3], dtype=tf.int32)
        length_logit = tf.constant([5], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)
        logproba = logit_to_logproba(logit=logit, axis=2)

        loss_session = SimplifiedCtcLossData(
            logprobas=logproba,
            # logits=logit,
            labels=label,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assert_tensors_almost_equal(
            tf.exp(loss_session.alpha), tf.exp(loss_session.gamma[:, 0, 0]), places=None
        )

    def test_gamma_symmetry(self):
        """Test for the symmetry of the gamma tensor."""
        input_dict = generate_ctc_loss_inputs(
            max_logit_length=4, batch_size=1, random_seed=0, num_tokens=3, blank_index=0
        )

        loss_session = SimplifiedCtcLossData(
            labels=input_dict["labels"],
            logprobas=input_dict["logprobas"],
            label_length=input_dict["label_length"],
            logit_length=input_dict["logit_length"],
            blank_index=0,
        )
        transposed_hessian = tf.transpose(loss_session.hessian, perm=[0, 3, 4, 1, 2])

        self.assert_tensors_almost_equal(
            first=tf.exp(transposed_hessian),
            second=tf.exp(loss_session.hessian),
            places=6,
        )

    def test_second_derivative_shape(self):
        """Test for the shape of the second derivative."""
        batch_size = 2
        num_tokens = 3
        max_logit_length = 4
        blank_index = 0
        input_dict = generate_ctc_loss_inputs(
            max_logit_length=max_logit_length,
            batch_size=batch_size,
            random_seed=0,
            num_tokens=num_tokens,
            blank_index=blank_index,
        )
        logprobas = input_dict["logprobas"]

        def func(logprobas):
            return ctc_loss_from_logproba(
                labels=input_dict["labels"],
                logprobas=logprobas,
                label_length=input_dict["label_length"],
                logit_length=input_dict["logit_length"],
                blank_index=blank_index,
                ctc_loss_data_cls=SimplifiedCtcLossData,
            )

        with tf.GradientTape() as tape1:
            tape1.watch([logprobas])
            with tf.GradientTape() as tape2:
                tape2.watch([logprobas])
                loss = func(logprobas)
            gradient_analytic = tape2.gradient(loss, sources=logprobas)

        hessian_analytic = tape1.batch_jacobian(gradient_analytic, source=logprobas)

        self.assertEqual(
            [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens],
            list(hessian_analytic.shape),
        )

    def test_hessian_vs_finite_difference(self):
        """A test for the second derivative of the loss."""
        input_dict = generate_ctc_loss_inputs(
            max_logit_length=4, batch_size=2, random_seed=0, num_tokens=2, blank_index=0
        )
        logits = input_dict["logits"]

        def gradient_fn(logits):
            with tf.GradientTape() as tape:
                tape.watch([logits])
                loss = tf.reduce_sum(
                    simplified_ctc_loss(
                        labels=input_dict["labels"],
                        logits=logits,
                        label_length=input_dict["label_length"],
                        logit_length=input_dict["logit_length"],
                        blank_index=0,
                    )
                )
            gradient = tape.gradient(loss, sources=logits)
            # shape: [batch_size, logit_length, num_tokens]
            return gradient

        hessain_numerical = finite_difference_batch_jacobian(
            func=gradient_fn, x=logits, epsilon=1e-4
        )
        # shape: [batch_size, logit_length, num_tokens, logit_length, num_tokens]

        with tf.GradientTape() as tape:
            tape.watch([logits])
            gradient = gradient_fn(logits)

        hessain_analytic = tape.batch_jacobian(gradient, source=logits)

        self.assert_tensors_almost_equal(hessain_numerical, hessain_analytic, 2)

    def test_readme_example(self):
        """Test for the example from the README."""
        batch_size = 2
        num_token = 3  # = 2 tokens + blank
        logit_length = 5
        labels = tf.constant([[1, 2, 2, 1], [1, 2, 1, 0]], dtype=tf.int32)
        label_length = tf.constant([4, 3], dtype=tf.int32)
        logits = tf.zeros(shape=[batch_size, logit_length, num_token], dtype=tf.float32)
        logit_length = tf.constant([logit_length, logit_length - 1], dtype=tf.int32)

        with tf.GradientTape(
            persistent=True
        ) as tape1:  # persistent=True is for experimental_use_pfor=False below
            tape1.watch([logits])
            with tf.GradientTape() as tape2:
                tape2.watch([logits])
                loss = tf.reduce_sum(
                    classic_ctc_loss(
                        labels=labels,
                        logits=logits,
                        label_length=label_length,
                        logit_length=logit_length,
                        blank_index=0,
                    )
                )
            gradient = tape2.gradient(loss, sources=logits)
        _ = tape1.batch_jacobian(gradient, source=logits, experimental_use_pfor=False)
        # experimental_use_pfor=False is needed to avoid a bag in tf.batch_jacobian and tf.jacobian
        # occurred in TensorFlow Version 2.4.1

    def test_second_gradient_autograph(self):
        """Test for the second derivative of the loss in autograph mode."""
        batch_size = 2
        num_tokens = 3
        max_logit_length = 4
        blank_index = 0
        input_dict = generate_ctc_loss_inputs(
            max_logit_length=max_logit_length,
            batch_size=batch_size,
            random_seed=0,
            num_tokens=num_tokens,
            blank_index=blank_index,
        )
        logits = input_dict["logits"]

        @tf.function
        def func(logits):
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([logits])
                with tf.GradientTape() as tape2:
                    tape2.watch([logits])
                    loss = tf.reduce_sum(
                        classic_ctc_loss(
                            labels=input_dict["labels"],
                            logits=logits,
                            label_length=input_dict["label_length"],
                            logit_length=input_dict["logit_length"],
                            blank_index=blank_index,
                        )
                    )
                gradient = tape2.gradient(loss, sources=logits)
            hessian = tape1.batch_jacobian(
                gradient, source=logits, experimental_use_pfor=False
            )

            return hessian

        hessian = func(logits=logits)

        self.assertEqual(
            [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens],
            list(hessian.shape),
        )
