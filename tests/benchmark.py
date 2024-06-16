"""Benchmark for CTC losses implementations."""

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
import logging
from datetime import datetime
from typing import Any, Callable, Dict
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tabulate import tabulate

from tests.common import generate_ctc_loss_inputs, tf_ctc_loss
from tf_seq2seq_losses.classic_ctc_loss import classic_ctc_loss
from tf_seq2seq_losses.simplified_ctc_loss import simplified_ctc_loss


class TestBenchmarkCtcLosses(unittest.TestCase):
    """Benchmark for CTC losses implementations."""

    def setUp(self) -> None:
        """Set up the test environment."""
        logging.getLogger().setLevel(logging.INFO)
        self.batch_size = 256
        self.num_tokens = 32
        self.max_logit_length = 255
        self.random_seed = 0
        self.blank_index = 0
        input_dict = generate_ctc_loss_inputs(
            batch_size=self.batch_size,
            max_logit_length=self.max_logit_length,
            random_seed=self.random_seed,
            num_tokens=self.num_tokens,
            blank_index=self.blank_index,
        )
        self.logits = input_dict["logits"]
        self.labels = input_dict["labels"]
        self.label_length = input_dict["label_length"]
        self.logit_length = input_dict["logit_length"]

    def test_all_gradients_benchmark(self) -> None:
        """Test all gradient benchmarks."""
        logging.info("Loss gradient benchmark:")

        performance_test_results = pd.DataFrame(
            columns=["mean processing time (s)", "std"]
        )
        performance_test_results.index.name = "implementation"
        performance_test_results.loc["tensorflow.nn.ctc_loss"] = (
            self.benchmark_tf_ctc_loss_gradient()
        )
        performance_test_results.loc["classic_ctc_loss"] = (
            self.benchmark_classic_ctc_loss_gradient()
        )
        performance_test_results.loc["simple_ctc_loss"] = (
            self.benchmark_simple_ctc_loss_gradient()
        )

        result_table = tabulate(
            performance_test_results,
            headers=performance_test_results.columns,
            tablefmt="grid",
            floatfmt=("", ".3g", ".1g"),
        )
        logging.info(f"{result_table}")

    def test_all_forwards_benchmarks(self) -> None:
        """Test all forward benchmarks."""
        logging.info("Loss forward benchmark:")

        performance_test_results = pd.DataFrame(
            columns=["mean processing time (s)", "std"]
        )
        performance_test_results.index.name = "implementation"
        performance_test_results.loc["tensorflow.nn.ctc_loss"] = (
            self.benchmark_tf_ctc_loss_forward()
        )
        performance_test_results.loc["classic_ctc_loss"] = (
            self.benchmark_classic_ctc_loss_forward()
        )
        performance_test_results.loc["simple_ctc_loss"] = (
            self.benchmark_simplified_ctc_loss_forward()
        )

        result_table = tabulate(
            performance_test_results,
            headers=performance_test_results.columns,
            tablefmt="grid",
            floatfmt=("", ".3g", ".1g"),
        )
        logging.info(f"\n{result_table}")

    def benchmark_tf_ctc_loss_gradient(self) -> Dict[str, Any]:
        """Benchmark TensorFlow CTC loss gradient."""
        return self.evaluate(
            func=self._gradient_graph(loss_fn=tf_ctc_loss),
            num_total_steps=10,
            num_warm_up_steps=3,
            name="tf_ctc_loss",
        )

    def benchmark_classic_ctc_loss_gradient(self) -> Dict[str, Any]:
        """Benchmark classic CTC loss gradient."""
        return self.evaluate(
            func=self._gradient_graph(loss_fn=classic_ctc_loss),
            num_total_steps=10,
            num_warm_up_steps=3,
            name="classic_ctc_loss",
        )

    def benchmark_simple_ctc_loss_gradient(self) -> Dict[str, Any]:
        """Benchmark simplified CTC loss gradient."""
        return self.evaluate(
            func=self._gradient_graph(loss_fn=simplified_ctc_loss),
            num_total_steps=10,
            num_warm_up_steps=3,
            name="simple_ctc_loss",
        )

    def benchmark_tf_ctc_loss_forward(self) -> Dict[str, Any]:
        """Benchmark TensorFlow CTC loss forward."""
        return self.evaluate(
            func=tf.function(tf_ctc_loss),
            num_total_steps=10,
            num_warm_up_steps=3,
            name="tensorflow.nn.ctc_loss",
        )

    def benchmark_classic_ctc_loss_forward(self) -> Dict[str, Any]:
        """Benchmark classic CTC loss forward."""
        return self.evaluate(
            func=tf.function(classic_ctc_loss),
            num_total_steps=10,
            num_warm_up_steps=3,
            name="classic_ctc_loss",
        )

    def benchmark_simplified_ctc_loss_forward(self) -> Dict[str, Any]:
        """Benchmark simplified CTC loss forward."""
        return self.evaluate(
            func=tf.function(simplified_ctc_loss),
            num_total_steps=10,
            num_warm_up_steps=3,
            name="simplified_ctc_loss",
        )

    def _forward_graph(
        self, loss_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], Any]
    ):
        """Create a graph for the forward calculation."""

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, None, self.num_tokens], dtype=tf.float32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
            ]
        )
        def func_graph(labels, logits, label_length, logit_length):
            return loss_fn(labels, logits, label_length, logit_length)

        return func_graph

    def _gradient_graph(
        self, loss_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]
    ) -> Any:
        """Create a graph for the gradient calculation."""

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, None, self.num_tokens], dtype=tf.float32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
            ]
        )
        def func_graph(labels, logits, label_length, logit_length):
            with tf.GradientTape() as tape:
                tape.watch([logits])
                output = loss_fn(labels, logits, label_length, logit_length)
                loss = tf.reduce_mean(output)
            gradient = tape.gradient(loss, sources=logits)
            return loss, gradient

        return func_graph

    def evaluate(
        self, func: Callable, num_total_steps: int, num_warm_up_steps: int, name: str
    ) -> Dict[str, Any]:
        """Evaluate the function."""
        for _ in tqdm(
            range(num_warm_up_steps),
            desc=f"{name}: warming up",
            total=num_warm_up_steps,
        ):
            func(
                labels=self.labels,
                logits=self.logits,
                label_length=self.label_length,
                logit_length=self.logit_length,
            )
        timestamps = [datetime.now()]
        for _ in tqdm(
            range(num_total_steps), desc=f"{name}: evaluation", total=num_total_steps
        ):
            func(self.labels, self.logits, self.label_length, self.logit_length)
            timestamps.append(datetime.now())

        time_intervals = np.array(
            [
                (t2 - t1).total_seconds()
                for t1, t2 in zip(timestamps[:-1], timestamps[1:])
            ]
        )

        return {
            "mean processing time (s)": np.mean(time_intervals),
            "std": np.std(time_intervals),
        }
