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

import numpy as np
import tensorflow as tf

from tests.common import generate_ctc_loss_inputs
from tests.test_ctc_losses import TestCtcLoss
from tf_seq2seq_losses.classic_ctc_loss import ClassicCtcLossData, classic_ctc_loss
from tf_seq2seq_losses.tools import finite_difference_gradient


class TestClassicCtcLoss(TestCtcLoss):
    def test_basic_case(self):
        logits = tf.math.log(tf.constant([[[0, 1, 0]]], dtype=tf.float32))
        labels = tf.constant([[1]], dtype=tf.int32)
        length_label = tf.constant([1], dtype=tf.int32)
        length_logit = tf.constant([1], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)

        loss_data = ClassicCtcLossData(
            logits=logits,
            labels=labels,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assertTensorsAlmostEqual(
            tf.constant([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=tf.float32),
            tf.exp(loss_data.alpha),
            places=None
        )
        self.assertTensorsAlmostEqual(
            tf.constant([[[1, 1], [0, 1]], [[0, 0], [1, 1]]], dtype=tf.float32),
            tf.exp(loss_data.beta),
            places=None
        )
        self.assertEqual(0., loss_data.log_loss.numpy()[0])

    def test_closed_state(self):
        logit = tf.math.log(tf.constant([[[0, 1, 0], [1, 0, 0]]], dtype=tf.float32))
        label = tf.constant([[1]], dtype=tf.int32)
        length_label = tf.constant([1], dtype=tf.int32)
        length_logit = tf.constant([2], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)

        loss_data = ClassicCtcLossData(
            logits=logit,
            labels=label,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assertTensorsAlmostEqual(
            tf.constant([[[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[0, 0], [1, 0]]]], dtype=tf.float32),
            tf.exp(loss_data.alpha),
            places=None
        )
        self.assertTensorsAlmostEqual(
            tf.constant([[[[1, 1], [0, 1]], [[0, 0], [1, 1]], [[0, 0], [1, 1]]]], dtype=tf.float32),
            tf.exp(loss_data.beta),
            places=None
        )
        self.assertEqual(0., loss_data.log_loss.numpy()[0])

    def test_classic_loss_simple_case(self):
        logit = tf.math.log(tf.constant([[[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]]], dtype=tf.float32))
        label = tf.constant([[1, 2, 2, 1]], dtype=tf.int32)
        length_label = tf.constant([4], dtype=tf.int32)
        length_logit = tf.constant([5], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)

        loss_data = ClassicCtcLossData(
            logits = logit,
            labels = label,
            logit_length = length_logit,
            label_length = length_label,
            blank_index = blank_index,
        )

        self.assertLess(loss_data.log_loss.numpy()[0], 1e-6)

    def test_alpha_beta_sum(self):
        input_dict = \
            generate_ctc_loss_inputs(max_logit_length=6, batch_size=1, random_seed=1, num_tokens=5, blank_index=0)
        loss_data = ClassicCtcLossData(**input_dict)

        # Sums along U of products alpha * beta that is supposed to be equal to the loss function
        sums = tf.reduce_logsumexp(loss_data.alpha + loss_data.beta, axis=[2, 3])

        # We verify that the values of the sums a equal to the loss up to a sign.
        self.assertTensorsAlmostEqual(
            first = -tf.expand_dims(loss_data.log_loss, 1),
            second = sums,
            places = 6,
        )

    def test_length_two_case(self):
        batch_size = 2
        num_tokens = 3
        blank_index = 0
        logits = tf.zeros(shape=[batch_size, 2, num_tokens])
        labels = tf.constant([[1, 2], [1, 2]])
        label_length = tf.constant([2, 1], dtype=tf.int32)
        logit_length = tf.constant([2, 2], dtype=tf.int32)

        loss_data = ClassicCtcLossData(
            logits = logits,
            labels = labels,
            logit_length = logit_length,
            label_length = label_length,
            blank_index = blank_index,
        )

        self.assertAlmostEqual(-np.log(1/3 * 1/3), loss_data.log_loss.numpy()[0], 6)
        self.assertAlmostEqual(-np.log(3 * 1/3 * 1/3), loss_data.log_loss.numpy()[1], 6)
        self.assertTensorsAlmostEqual(
            first = tf.constant([[[0., 1., 0.], [0., 0., 1.]],  [[1/3, 2/3, 0.], [1/3, 2/3, 0.]]]),
            second = tf.exp(loss_data.logarithmic_log_proba_gradient),
            places = 6
        )
        self.assertTensorsAlmostEqual(
            first = tf.constant([[[1/3, -2/3, 1/3], [1/3, 1/3, -2/3]], [[0., -1/3, 1/3], [0., -1/3, 1/3]]]),
            second = loss_data.gradient,
            places = 6
        )

    def test_repeated_token_case(self):
        num_tokens = 3
        blank_index = 0
        logits = tf.zeros(shape=[1, 3, num_tokens])
        labels = tf.constant([[1, 1]])
        label_length = tf.constant([2], dtype=tf.int32)
        logit_length = tf.constant([3], dtype=tf.int32)

        loss_session = ClassicCtcLossData(
            logits = logits,
            labels = labels,
            logit_length = logit_length,
            label_length = label_length,
            blank_index = blank_index,
        )

        # Label "aa" corresponds to a single paths: "a_a" with probability 3 ** -3
        self.assertAlmostEqual(np.log(3 ** 3), loss_session.log_loss.numpy()[0], 6)

    def test_repeated_token_case_2(self):
        num_tokens = 3
        blank_index = 0
        logits = tf.zeros(shape=[1, 3, num_tokens])
        labels = tf.constant([[1]])
        label_length = tf.constant([1], dtype=tf.int32)
        logit_length = tf.constant([3], dtype=tf.int32)

        loss_session = ClassicCtcLossData(
            logits = logits,
            labels = labels,
            logit_length = logit_length,
            label_length = label_length,
            blank_index = blank_index,
        )

        # Label "a" corresponds to 6 paths: "a__", "_a_", "__a", "aa_", "_aa", and "aaa" with equal probability 3 ** -3
        self.assertAlmostEqual(np.log(3 ** 3 / 6), loss_session.log_loss.numpy()[0], 6)

    def test_wrong_prediction_case(self):
        logit = tf.constant([[[0, 0, 1]]], dtype=tf.float32) * 100
        label = tf.constant([[1]], dtype=tf.int32)
        length_label = tf.constant([1], dtype=tf.int32)
        length_logit = tf.constant([1], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)

        loss_data = ClassicCtcLossData(
            logits=logit,
            labels=label,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assertTensorsAlmostEqual(
            tf.constant([[[0., -1., 1.]]], dtype=tf.float32),
            loss_data.gradient,
            places=None
        )
        self.assertEqual(100., loss_data.log_loss.numpy()[0])

    def test_zero_batch_size(self):
        logits = tf.zeros(shape=[0, 4, 3], dtype=tf.float32)
        labels = tf.zeros(shape=[0, 2], dtype=tf.int32)
        label_length = tf.zeros(shape=[0], dtype=tf.int32)
        logit_length = tf.zeros(shape=[0], dtype=tf.int32)

        @tf.function
        def func():
            with tf.GradientTape() as tape:
                tape.watch([logits])
                loss_samplewise = classic_ctc_loss(labels, logits, label_length, logit_length, 0)
                loss = tf.reduce_sum(loss_samplewise)
                gradient = tape.gradient(loss, sources=logits)
            return loss_samplewise, gradient

        loss_samplewise, gradient = func()

        self.assertEqual([0], loss_samplewise.shape)
        self.assertEqual([0, 4, 3], gradient.shape)

    def test_compare_forward_with_tf_implementation(self):
        input_dict = \
            generate_ctc_loss_inputs(max_logit_length=20, batch_size=8, random_seed=0, num_tokens=8, blank_index=0)
        tf_ctc_loss = tf.nn.ctc_loss(
            labels = input_dict["labels"],
            logits = input_dict["logits"],
            label_length = input_dict["label_length"],
            logit_length = input_dict["logit_length"],
            logits_time_major = False,
            blank_index = 0,
        )
        loss_session = ClassicCtcLossData(**input_dict)

        ctc_loss = loss_session.log_loss

        self.assertTensorsAlmostEqual(tf_ctc_loss, ctc_loss, 5)

    def test_compare_gradient_with_tf_implementation(self):
        input_dict = \
            generate_ctc_loss_inputs(max_logit_length=64, batch_size=8, random_seed=0, num_tokens=10, blank_index=0)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([input_dict["logits"]])
            tf_loss = tf.nn.ctc_loss(
                labels = input_dict["labels"],
                logits = input_dict["logits"],
                label_length = input_dict["label_length"],
                logit_length = input_dict["logit_length"],
                logits_time_major = False,
                blank_index = 0,
            )
            testing_loss = classic_ctc_loss(
                labels = input_dict["labels"],
                logits = input_dict["logits"],
                label_length = input_dict["label_length"],
                logit_length = input_dict["logit_length"],
                blank_index = 0,
            )
            tf_version_gradient = tape.gradient(tf_loss, input_dict["logits"])
            classic_version_gradient = tape.gradient(testing_loss, input_dict["logits"])

        self.assertTensorsAlmostEqual(tf_version_gradient, classic_version_gradient, 4)

    def test_gradient_vs_finite_difference(self):
        input_dict = \
            generate_ctc_loss_inputs(max_logit_length=16, batch_size=1, random_seed=0, num_tokens=4, blank_index=0)
        logits = input_dict["logits"]
        def loss_fn(logit):
            return tf.reduce_sum(classic_ctc_loss(
                labels = input_dict["labels"],
                logits = tf.expand_dims(logit, 0),
                label_length = input_dict["label_length"],
                logit_length = input_dict["logit_length"],
                blank_index = 0,
            ))
        gradient_numerical = finite_difference_gradient(
            func=lambda logits: tf.vectorized_map(fn=loss_fn, elems=logits),
            x=logits,
            epsilon=1e-3
        )

        with tf.GradientTape() as tape:
            tape.watch([logits])
            loss = loss_fn(logits[0])
            gradient_analytic = tape.gradient(loss, sources=logits)

        self.assertTensorsAlmostEqual(gradient_numerical, gradient_analytic, 2)

    def test_readme_example(self):
        batch_size = 1
        num_token = 3
        logit_length = 5
        loss = classic_ctc_loss(
            labels=tf.constant([[1, 2, 2, 1]], dtype=tf.int32),
            logits=tf.zeros(shape=[batch_size, logit_length, num_token], dtype=tf.float32),
            label_length=tf.constant([4], dtype=tf.int32),
            logit_length=tf.constant([logit_length], dtype=tf.int32),
            blank_index=0,
        )