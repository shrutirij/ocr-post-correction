"""Constants for all the OCR post-correction experiments with the multisource model.

These can be modified for different languages/settings as needed.

Author: Shruti Rijhwani
Contact: srijhwan@cs.cmu.edu

Please cite:
OCR Post Correction for Endangered Language Texts (EMNLP 2020)
https://www.aclweb.org/anthology/2020.emnlp-main.478/
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
