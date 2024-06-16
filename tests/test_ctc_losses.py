"""Tests for the classic CTC loss."""

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
from typing import Optional
import numpy as np
import tensorflow as tf


class TestCtcLoss(unittest.TestCase):
    """Base class for loss tests."""

    def assert_tensors_almost_equal(
        self, first: tf.Tensor, second: tf.Tensor, places: Optional[int]
    ):
        """Asserts that two tensorflow tensors are almost equal (against L infinity norm).

        Args:
            first:  first instance to compare
            second: second instance to compare
            places: number of decimal places to compare

        Returns:    None
        """
        if places is None:
            self.assertTrue(tf.reduce_all(first == second).numpy())
        else:
            self.assertAlmostEqual(
                first=0,
                second=tf.norm(first - second, ord=np.inf).numpy(),
                places=places,
            )
