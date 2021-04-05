"""Utility classes for the OCR post-correction model.

This file includes:
(1) Class ErrorMetrics for computing the post-correction performance (character and word error rates);
(2) Class CharVocab for the character vocabulary used by the sequence-to-sequence model;
(3) Class DataReader for preparing the data to be used for training and evaluation;
(4) Class Hypothesis for the hypotheses used in beam search.

Copyright (c) 2021, Shruti Rijhwani
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


import argparse
from collections import defaultdict
import re
import editdistance as ed
import math
from constants import UNK, EOS
import dynet as dy
import unicodedata
import logging
import json


class ErrorMetrics:
    def calculate_metrics(
        self, predicted_text, transcript, p_gens_all, output_file, write_pgens
    ):
        cer = ed.eval(predicted_text, transcript) / float(len(transcript))
        pred_spl = predicted_text.split()
        transcript_spl = transcript.split()
        wer = ed.eval(pred_spl, transcript_spl) / float(len(transcript_spl))
        if output_file:
            output_file.write(str(predicted_text) + "\n")
            if write_pgens:
                output_file.write("\n")
                for p_gens in p_gens_all:
                    output_file.write(
                        " ".join(["{0:.2f}".format(p) for p in p_gens]) + "\n"
                    )
        return cer, wer

    def get_average_cer(self, model, data, output_file, write_pgens=False):
        outputs = []
        p_gens_all = []
        transcripts = []
        for src1, src2, tgt in data:
            dy.renew_cg()
            output, p_gens = model.generate_beam(src1, src2)
            outputs.append(output)
            p_gens_all.append(p_gens)
            transcripts.append(
                "".join([model.tgt_vocab.int2str(idx) for idx in tgt[1:-1]])
            )
        cur_cer, cur_wer = self.calculate_metrics(
            "\n".join(outputs),
            "\n".join(transcripts),
            p_gens_all=p_gens_all,
            output_file=output_file,
            write_pgens=write_pgens,
        )
        return cur_cer, cur_wer


class CharVocab:
    def __init__(self, filepaths, lookup_json=None):
        if lookup_json:
            self.lookup = json.load(open(lookup_json, "r", encoding="utf8"))
        else:
            self.lookup = defaultdict(lambda: len(self.lookup))
            self.lookup[UNK]
            self.lookup[EOS]

            for filename in filepaths:
                for c in open(filename, encoding="utf8").read():
                    if c not in self.lookup:
                        self.lookup[c]

        self.reverse_lookup = {v: k for k, v in self.lookup.items()}

    def char2int(self, input_char):
        if input_char in self.lookup:
            return self.lookup[input_char]
        return self.lookup[UNK]

    def str2int(self, input_string):
        if input_string == EOS:
            return self.lookup[EOS]
        if input_string == UNK:
            return self.lookup[UNK]
        return [self.char2int(c) for c in input_string]

    def int2str(self, input_id):
        if input_id in self.reverse_lookup:
            return self.reverse_lookup[input_id]
        return -1

    def length(self):
        return len(self.lookup)

    def get_lookup(self):
        return self.lookup


class Hypothesis:
    def __init__(
        self, text_list, decoder_state, c1_t, c2_t, prev_coverage, score, p_gens
    ):
        self.decoder_state = decoder_state
        self.text_list = text_list
        self.score = score
        self.c1_t = c1_t
        self.c2_t = c2_t
        self.prev_coverage = prev_coverage
        self.p_gens = p_gens


class DataReader:
    def preprocess(self, text):
        preprocessed = " ".join(text.strip().split())
        return preprocessed

    def read_single_source_data(self, filen, vocab):
        data = []
        lines = open(filen, encoding="utf8").readlines()
        for line in lines:
            idxs = (
                [vocab.str2int(EOS)]
                + vocab.str2int(self.preprocess(line))
                + [vocab.str2int(EOS)]
            )
            data.append(idxs)
        return data

    def read_parallel_data(self, model, src1_path, src2_path, tgt_path):
        if not src2_path:
            src2_path = src1_path
        data = []

        src1_lines = open(src1_path, encoding="utf8").readlines()
        src2_lines = open(src2_path, encoding="utf8").readlines()
        tgt_lines = open(tgt_path, encoding="utf8").readlines()

        if len(src1_lines) != len(src2_lines) or len(src1_lines) != len(tgt_lines):
            logging.info(
                "Unequal lines in: {} {} {}".format(src1_path, src2_path, tgt_path)
            )
            return []

        for src1_line, src2_line, tgt_line in zip(src1_lines, src2_lines, tgt_lines):
            if (
                (not src1_line.strip())
                or (not src2_line.strip())
                or (not tgt_line.strip())
            ):
                continue
            src1_enc = (
                [model.src1_vocab.str2int(EOS)]
                + model.src1_vocab.str2int(self.preprocess(src1_line))
                + [model.src1_vocab.str2int(EOS)]
            )
            src2_enc = (
                [model.src2_vocab.str2int(EOS)]
                + model.src2_vocab.str2int(self.preprocess(src2_line))
                + [model.src2_vocab.str2int(EOS)]
            )
            tgt_enc = (
                [model.tgt_vocab.str2int(EOS)]
                + model.tgt_vocab.str2int(self.preprocess(tgt_line))
                + [model.tgt_vocab.str2int(EOS)]
            )
            data.append((src1_enc, src2_enc, tgt_enc))
        return data

    def read_test_data(self, model, src1, src2):
        if not src2:
            src2 = src1

        data = []
        src1_lines = open(src1, encoding="utf8").readlines()
        src2_lines = open(src2, encoding="utf8").readlines()
        assert len(src1_lines) == len(src2_lines)

        for src1_line, src2_line in zip(src1_lines, src2_lines):
            if (not src1_line.strip()) or (not src2_line.strip()):
                data.append(([], []))
                continue
            src1_enc = (
                [model.src1_vocab.str2int(EOS)]
                + model.src1_vocab.str2int(self.preprocess(src1_line))
                + [model.src1_vocab.str2int(EOS)]
            )
            src2_enc = (
                [model.src2_vocab.str2int(EOS)]
                + model.src2_vocab.str2int(self.preprocess(src2_line))
                + [model.src2_vocab.str2int(EOS)]
            )
            data.append((src1_enc, src2_enc))
        return data
