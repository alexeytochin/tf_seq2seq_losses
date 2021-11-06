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

"""
tf-seq2seq-losses

Tensorflow implementations for seq2seq Machine Learning model loss functions
"""
from tf_seq2seq_losses.classic_ctc_loss import classic_ctc_loss
from tf_seq2seq_losses.simplified_ctc_loss import simplified_ctc_loss


__version__ = "0.1.1"
__author__ = 'Alexey Tochin'
__all__ = ["classic_ctc_loss", "simplified_ctc_loss"]