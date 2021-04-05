"""Module for setting the configuration of the experiment.

This class contains functions to parse the input options set by the user.
It also contains logging, outputs, and model saving information.

Copyright (c) 2021, Shruti Rijhwani
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


import argparse
import logging
import os


class SetConfig:
    def __init__(self, sys_args):
        self.args = self.parse_config(sys_args)

    def parse_config(self, sys_args):
        parser = argparse.ArgumentParser()

        # model params
        parser.add_argument(
            "--single",
            action="store_true",
            help="Enables single-source model. Default is multisource.",
        )
        parser.add_argument(
            "--beam_size",
            type=int,
            default=4,
            help="Beam size for generating outputs with beam search.",
        )
        parser.add_argument(
            "--pointer_gen",
            action="store_true",
            help="Enables copy mechanism (recommended).",
        )
        parser.add_argument(
            "--coverage",
            action="store_true",
            help="Enables coverage mechanism and coverage loss (recommended).",
        )
        parser.add_argument(
            "--diag_loss",
            type=int,
            default=-1,
            help="Enables diagonal attention loss (recommended).",
        )
        parser.add_argument(
            "--load_model",
            default=None,
            help="Saved model to load before training or testing.",
        )
        parser.add_argument(
            "--vocab_folder",
            help="Location of stored character vocabularies to use with the model.",
        )

        # experiment params
        parser.add_argument(
            "--pretrain_only",
            action="store_true",
            help="Enables pretraining only. Useful when using the same pretrained model for many experiments.",
        )
        parser.add_argument(
            "--train_only",
            action="store_true",
            help="Enables training only. Useful when setting up experiments with different combinations of datasets and/or pretrained models.",
        )
        parser.add_argument(
            "--testing",
            action="store_true",
            help="Enables testing with a trained model.",
        )
        parser.add_argument("--model_name", help="Model name for saving to disk.")

        # pretrain params
        parser.add_argument(
            "--pretrain_src1",
            help="Source 1 for pretraining (endangered language first pass OCR).",
        )
        parser.add_argument(
            "--pretrain_src2",
            help="Source 2 for pretraining (translation first pass OCR). Do not use for single-source model.",
        )
        parser.add_argument(
            "--pretrain_tgt", help="Denoised target outputs for pretraining."
        )
        parser.add_argument(
            "--pretrain_enc",
            action="store_true",
            help="Enables pretraining the encoders.",
        )
        parser.add_argument(
            "--pretrain_dec",
            action="store_true",
            help="Enables pretraining the decoders.",
        )
        parser.add_argument(
            "--pretrain_s2s",
            action="store_true",
            help="Enables pretraining the seq2seq model.",
        )
        parser.add_argument(
            "--pretrain_epochs",
            type=int,
            default=10,
            help="Number of epochs for pretraining.",
        )

        # parallel files for training -- one sentence or paragraph per line
        parser.add_argument(
            "--train_src1",
            help="Source 1 for training (endangered language first pass OCR).",
        )
        parser.add_argument(
            "--train_src2",
            help="Source 2 for training (translation first pass OCR). Do not use for single-source model.",
        )
        parser.add_argument(
            "--train_tgt", help="Manually corrected target transcriptions for training."
        )
        parser.add_argument("--dev_src1", help="Development set source 1.")
        parser.add_argument(
            "--dev_src2",
            help="Development set source 2. Do not use for single-source model.",
        )
        parser.add_argument(
            "--dev_tgt",
            help="Development set manually corrected target transcriptions.",
        )

        # test params
        parser.add_argument("--test_src1", help="Test input source 1.")
        parser.add_argument(
            "--test_src2",
            help="Test input source 2. Do not use for single-source model.",
        )
        parser.add_argument(
            "--test_tgt",
            help="Optional target transcriptions for testing. Use only when targets are available and CER/WER metrics are desired.",
        )

        # dynet params
        parser.add_argument(
            "--dynet-autobatch", help="Enables automatic minibatching with dynet."
        )
        parser.add_argument("--dynet-mem", help="Memory available to dynet.")
        parser.add_argument(
            "--dynet-gpu", action="store_true", help="Enables GPU use with dynet."
        )

        # experiment output folder
        parser.add_argument(
            "--output_folder",
            help="Output folder for experiment logs, output files, and saved models.",
        )

        args = parser.parse_args(sys_args)

        output_folder = args.output_folder

        if args.pretrain_only:
            filename = args.model_name
            log_filename = output_folder + "pretrain_logs/" + filename + ".log"
            self.model_name = output_folder + "pretrain_models/" + filename
            self.output_name = None
        elif args.testing:
            log_filename = None
            self.output_name = (
                output_folder + "outputs/" + args.load_model.split("/")[-1]
            )
            self.model_name = None
        else:
            filename = args.model_name
            log_filename = output_folder + "train_logs/" + filename + ".log"
            self.model_name = output_folder + "models/" + filename
            self.output_name = output_folder + "debug_outputs/" + filename

        if log_filename:
            logging.basicConfig(
                filename=log_filename,
                level=logging.INFO,
                format="%(message)s",
                filemode="w",
            )
        return args
