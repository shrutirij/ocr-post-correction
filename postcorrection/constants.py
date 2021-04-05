"""Constants for all the OCR post-correction experiments with the multisource model.

These can be modified for different languages/settings as needed.

Copyright (c) 2021, Shruti Rijhwani
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


LSTM_NUM_OF_LAYERS = 1
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
ATTENTION_SIZE = 256
UNK = "<unk>"
EOS = "<eos>"
COV_LOSS_WEIGHT = 1.0
DIAG_LOSS_WEIGHT = 1.0
EPOCHS = 150
PATIENCE = 10
