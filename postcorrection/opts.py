"""[summary]

Author: Shruti Rijhwani
Contact: srijhwan@cs.cmu.edu

Please cite:
OCR Post Correction for Endangered Language Texts (EMNLP 2020)
https://www.aclweb.org/anthology/2020.emnlp-main.478/
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
        parser.add_argument("--single", action="store_true")
        parser.add_argument("--beam_size", type=int, default=4)
        parser.add_argument("--pointer_gen", action="store_true")
        parser.add_argument("--coverage", action="store_true")
        parser.add_argument("--diag_loss", type=int, default=-1)
        parser.add_argument("--load_model", default=None)
        parser.add_argument("--vocab_folder")

        # experiment params
        parser.add_argument("--pretrain_only", action="store_true")
        parser.add_argument("--train_only", action="store_true")
        parser.add_argument("--testing", action="store_true")
        parser.add_argument("--model_name")

        # pretrain params
        parser.add_argument("--pretrain_src1")
        parser.add_argument("--pretrain_src2")
        parser.add_argument("--pretrain_tgt")
        parser.add_argument("--pretrain_enc", action="store_true")
        parser.add_argument("--pretrain_dec", action="store_true")
        parser.add_argument("--pretrain_s2s", action="store_true")
        parser.add_argument("--pretrain_epochs", type=int, default=10)

        # parallel files for training -- one sentence/paragraph per line
        parser.add_argument("--train_src1")
        parser.add_argument("--train_src2")
        parser.add_argument("--train_tgt")
        parser.add_argument("--dev_src1")
        parser.add_argument("--dev_src2")
        parser.add_argument("--dev_tgt")

        # test params
        parser.add_argument("--test_src1")
        parser.add_argument("--test_src2")
        parser.add_argument("--test_tgt")

        # dynet params
        parser.add_argument("--dynet-autobatch")
        parser.add_argument("--dynet-mem")
        parser.add_argument("--dynet-gpu", action="store_true")

        # experiment output folder
        parser.add_argument("--output_folder")

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
