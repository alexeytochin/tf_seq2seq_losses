import numpy as np
import tensorflow as tf

from tests.common import generate_ctc_loss_inputs
from tests.test_ctc_losses import TestCtcLoss
from tf_seq2seq_losses.simplified_ctc_loss import simplified_ctc_loss, SimplifiedCtcLossData
from tf_seq2seq_losses.tools import finite_difference_gradient


class TestSimplifiedCtcLoss(TestCtcLoss):
    def test_simple_case(self):
        logits = tf.math.log(tf.constant([[[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]], dtype=tf.float32))
        labels = tf.constant([[1, 2, 1]], dtype=tf.int32)
        length_label = tf.constant([3], dtype=tf.int32)
        length_logit = tf.constant([5], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)

        loss_session = SimplifiedCtcLossData(
            logits=logits,
            labels=labels,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assertTrue(tf.reduce_all(
            tf.exp(loss_session.alpha) == tf.constant(
                [[[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]]]
        )).numpy())
        self.assertTrue(tf.reduce_all(
            tf.exp(loss_session.beta) == tf.constant(
                [[[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]]]
        )).numpy())
        self.assertLess(loss_session.log_loss.numpy()[0], 1e-6)

    def test_non_zero_blank_index(self):
        logit = tf.math.log(tf.constant([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]], dtype=tf.float32))
        label = tf.constant([[0, 2, 0]], dtype=tf.int32)
        length_label = tf.constant([3], dtype=tf.int32)
        length_logit = tf.constant([5], dtype=tf.int32)
        blank_index = tf.constant(1, dtype=tf.int32)

        loss_session = SimplifiedCtcLossData(
            logits=logit,
            labels=label,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assertLess(loss_session.log_loss.numpy()[0], 1e-6)

    def test_shorter_logit_and_label_length(self):
        logits = tf.math.log(tf.constant([[[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]], dtype=tf.float32))
        labels = tf.constant([[1, 0]], dtype=tf.int32)
        logit_length = tf.constant([3], dtype=tf.int32)
        label_length = tf.constant([1], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)

        loss_session = SimplifiedCtcLossData(
            logits=logits,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank_index=blank_index,
        )

        self.assertEqual([0], loss_session.log_loss.numpy().tolist())

    def test_label_length_bigger_then_logit_length(self):
        logits = tf.constant([[[0, 0, 0]]], dtype=tf.float32)
        labels = tf.constant([[1, 2]], dtype=tf.int32)
        logit_length = tf.constant([1], dtype=tf.int32)
        label_length = tf.constant([2], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)

        loss_session = SimplifiedCtcLossData(
            logits=logits,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank_index=blank_index,
        )

        self.assertEqual(np.inf, loss_session.log_loss.numpy()[0])
        self.assertEqual(np.zeros(shape=[1, 1, 3]).tolist(), loss_session.gradient.numpy().tolist())

    def test_large_loss(self):
        logit = tf.constant([[[1e+10, 0., 0.]]], dtype=tf.float32)
        label = tf.constant([[1]], dtype=tf.int32)
        length_label = tf.constant([1], dtype=tf.int32)
        length_logit = tf.constant([1], dtype=tf.int32)
        blank_index = tf.constant(0, dtype=tf.int32)

        loss_session = SimplifiedCtcLossData(
            logits=logit,
            labels=label,
            logit_length=length_logit,
            label_length=length_label,
            blank_index=blank_index,
        )

        self.assertEqual(1e+10, loss_session.log_loss.numpy()[0])
        self.assertEqual([[[1., -1., 0.]]], loss_session.gradient.numpy().tolist())

    def test_alpha_beta_sum(self):
        input_dict = \
            generate_ctc_loss_inputs(max_logit_length=6, batch_size=1, random_seed=1, num_tokens=5, blank_index=0)
        loss_session = SimplifiedCtcLossData(**input_dict)

        # Sums along U of products alpha * beta that is supposed to be equal to the loss function
        sums = tf.reduce_logsumexp(loss_session.alpha + loss_session.beta, axis=2)

        # We verify that the values of the sums a equal to the loss up to a sign.
        self.assertTensorsAlmostEqual(
            first=-tf.expand_dims(loss_session.log_loss, 1),
            second=sums,
            places=6,
        )

    def test_length_one(self):
        batch_size = 1
        num_tokens = 3
        blank_index = 0
        logits = tf.zeros(shape=[batch_size, 1, num_tokens])
        labels = tf.constant([[1]])
        label_length = tf.constant([1], dtype=tf.int32)
        logit_length = tf.constant([1], dtype=tf.int32)

        loss_session = SimplifiedCtcLossData(
            logits=logits,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank_index=blank_index,
        )

        self.assertTensorsAlmostEqual(np.log(num_tokens), loss_session.log_loss[0], 8)
        self.assertTensorsAlmostEqual(
            first=tf.constant([[[1/3, -2/3, 1/3]]]),
            second=loss_session.gradient,
            places=6
        )

    def test_length_two(self):
        batch_size = 1
        num_tokens = 3
        blank_index = 0
        logits = tf.zeros(shape=[batch_size, 2, num_tokens])
        labels = tf.constant([[1, 2]])
        label_length = tf.constant([2], dtype=tf.int32)
        logit_length = tf.constant([2], dtype=tf.int32)

        loss_session = SimplifiedCtcLossData(
            logits=logits,
            labels=labels,
            logit_length=logit_length,
            label_length=label_length,
            blank_index=blank_index,
        )

        self.assertTensorsAlmostEqual(2 * np.log(num_tokens), loss_session.log_loss[0], 8)
        self.assertTensorsAlmostEqual(
            first=tf.constant([[[1/3, -2/3, 1/3], [1/3, 1/3, -2/3]]]),
            second=loss_session.gradient,
            places=6
        )

    def test_gradient_with_final_difference(self):
        input_dict = \
            generate_ctc_loss_inputs(max_logit_length=4, batch_size=1, random_seed=0, num_tokens=3, blank_index=0)
        logits = input_dict["logits"]

        def loss_fn(logit):
            return tf.reduce_sum(simplified_ctc_loss(
                labels=input_dict["labels"],
                logits=tf.expand_dims(logit, 0),
                label_length=input_dict["label_length"],
                logit_length=input_dict["logit_length"],
                blank_index=0,
            ))

        gradient_numerical = finite_difference_gradient(
            func=lambda logits_: tf.vectorized_map(fn=loss_fn, elems=logits_),
            x=logits,
            epsilon=1e-5
        )

        with tf.GradientTape() as tape:
            tape.watch([logits])
            loss = loss_fn(logits[0])
        gradient_analytic = tape.gradient(loss, sources=logits)

        self.assertTensorsAlmostEqual(gradient_numerical, gradient_analytic, 1)

    def test_autograph(self):
        # We test that TesorFlow graph can be built form simplified_ctc_loss and its gradient function.
        loss_inputs = \
            generate_ctc_loss_inputs(max_logit_length=6, batch_size=2, random_seed=0, num_tokens=3, blank_index=0)

        @tf.function
        def func() -> tf.Tensor:
            with tf.GradientTape() as tape:
                tape.watch([loss_inputs["logits"]])
                output = simplified_ctc_loss(**loss_inputs)
                loss = tf.reduce_mean(output)
            gradient = tape.gradient(loss, sources=loss_inputs["logits"])
            return gradient

        # This should not raise an exception
        func()

    def test_zero_logit_length(self):
        logits = tf.zeros(shape=[1, 0, 3])
        labels = tf.constant([[1, 2]])
        label_length = tf.constant([2], dtype=tf.int32)
        logit_length = tf.constant([2], dtype=tf.int32)

        @tf.function
        def func():
            with tf.GradientTape() as tape:
                tape.watch([logits])
                output = simplified_ctc_loss(labels, logits, label_length, logit_length, 0)
                loss = tf.reduce_mean(output)
            gradient = tape.gradient(loss, sources=logits)
            return loss, gradient

        loss, gradient = func()

        self.assertEqual(np.inf, loss.numpy())
        self.assertEqual([1, 0, 3], list(gradient.shape))

    def test_zero_batch_size(self):
        logits = tf.zeros(shape=[0, 4, 3], dtype=tf.float32)
        labels = tf.zeros(shape=[0, 2], dtype=tf.int32)
        label_length = tf.zeros(shape=[0], dtype=tf.int32)
        logit_length = tf.zeros(shape=[0], dtype=tf.int32)

        @tf.function
        def func():
            with tf.GradientTape() as tape:
                tape.watch([logits])
                loss_samplewise = simplified_ctc_loss(labels, logits, label_length, logit_length, 0)
                loss = tf.reduce_sum(loss_samplewise)
            gradient = tape.gradient(loss, sources=logits)
            return loss_samplewise, gradient

        loss_samplewise, gradient = func()

        self.assertEqual([0], loss_samplewise.shape)
        self.assertEqual([0, 4, 3], gradient.shape)
