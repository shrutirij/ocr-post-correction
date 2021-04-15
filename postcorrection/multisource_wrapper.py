"""Wrapper class for the post-correction model functions.

It includes steps to load the vocabulary, pretrain, train, and test with the model.

Copyright (c) 2021, Shruti Rijhwani
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


import sys
import _dynet as dy
import argparse

from utils import ErrorMetrics, CharVocab, Hypothesis, DataReader
from seq2seq_trainer import Seq2SeqTrainer
from seq2seq_tester import Seq2SeqTester
from opts import SetConfig
from multisource_model import TwoSourceModel
from pretrain_handler import PretrainHandler

from collections import defaultdict
import re
import glob
import sys
import editdistance as ed
import math
import random
import logging
import numpy as np


if __name__ == "__main__":
    config = SetConfig(sys.argv[1:])

    src1_vocab = CharVocab(
        filepaths=None, lookup_json="{}/src1.json".format(config.args.vocab_folder)
    )
    src2_vocab = CharVocab(
        filepaths=None, lookup_json="{}/src2.json".format(config.args.vocab_folder)
    )
    tgt_vocab = CharVocab(
        filepaths=None, lookup_json="{}/tgt.json".format(config.args.vocab_folder)
    )

    model = TwoSourceModel(
        src1_vocab=src1_vocab,
        src2_vocab=src2_vocab,
        tgt_vocab=tgt_vocab,
        single=config.args.single,
        pointer_gen=config.args.pointer_gen,
        coverage=config.args.coverage,
        load_model=config.args.load_model,
        model_file=config.model_name,
        diag_loss=config.args.diag_loss,
        beam_size=config.args.beam_size,
        best_val_cer=1.0,
    )

    if not config.args.train_only and not config.args.testing:
        pretrainer = PretrainHandler(
            model,
            pretrain_src1=config.args.pretrain_src1,
            pretrain_src2=config.args.pretrain_src2,
            pretrain_tgt=config.args.pretrain_tgt,
            pretrain_enc=config.args.pretrain_enc,
            pretrain_dec=config.args.pretrain_dec,
            pretrain_model=config.args.pretrain_s2s,
            epochs=config.args.pretrain_epochs,
        )

    elif not config.args.pretrain_only and not config.args.testing:
        trainer = Seq2SeqTrainer(model, output_name=config.output_name)
        trainer.train_model(
            train_src1=config.args.train_src1,
            train_src2=config.args.train_src2,
            train_tgt=config.args.train_tgt,
            val_src1=config.args.dev_src1,
            val_src2=config.args.dev_src2,
            val_tgt=config.args.dev_tgt,
        )

    elif config.args.testing:
        tester = Seq2SeqTester(model, output_name=config.output_name)
        tester.test(
            src1=config.args.test_src1,
            src2=config.args.test_src2,
            tgt=config.args.test_tgt,
        )
