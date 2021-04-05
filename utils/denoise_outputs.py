"""Script to denoise first pass OCR outputs using a small amount of corrected data.

The small amount of manually corrected data is used to create probabilistic "denoising rules".
The rules are then applied to the input first pass files to automatically create "denoised outputs"

The denoised outputs are subsequently used to pretrain the post-correction model.

Usage:
python denoise_outputs.py --train_src1 [train_src] --train_tgt [train_tgt] --input [input_filename] --output [output_filename]

Copyright (c) 2021, Shruti Rijhwani
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


import argparse
import glob
import random
import Levenshtein
from collections import defaultdict


class Denoiser(object):
    def preprocess(self, text):
        preprocessed = " ".join(text.strip().split())
        return preprocessed

    def count_chars(self, predicted_text, char_counts):
        for c in predicted_text:
            char_counts[c] += 1
            char_counts["total"] += 1

    def error_distribution(self, src, tgt, errors):
        edits = Levenshtein.editops(src, tgt)
        for edit in edits:
            if edit[0] == "replace":
                errors[("replace", src[edit[1]], tgt[edit[2]])] += 1
            elif edit[0] == "delete":
                errors[("delete", src[edit[1]])] += 1
            elif edit[0] == "insert":
                errors[("insert", tgt[edit[2]])] += 1
            else:
                print(edit)

    def create_rules(self, src, tgt):
        src_lines = open(src, encoding="utf8").readlines()
        tgt_lines = open(tgt, encoding="utf8").readlines()
        assert len(src_lines) == len(tgt_lines)

        errors = defaultdict(lambda: 0)
        char_counts = defaultdict(lambda: 0)

        for src_line, tgt_line in zip(src_lines, tgt_lines):
            if (not src_line.strip()) or (not tgt_line.strip()):
                continue

            self.error_distribution(
                self.preprocess(src_line), self.preprocess(tgt_line), errors
            )
            self.count_chars(self.preprocess(src_line), char_counts)

        rules = {}
        for k, v in errors.items():
            if k[0] == "replace":
                rules[(k[0], k[1], k[2])] = v / char_counts[k[1]]
            elif k[0] == "delete":
                rules[(k[0], k[1], "")] = v / char_counts[k[1]]
            elif k[0] == "insert":
                rules[(k[0], k[1], "")] = v / char_counts["total"]
        return rules

    def denoise_file(self, rules, input_file, output_file):
        with open(input_file, "r", encoding="utf8") as f, open(
            output_file, "w", encoding="utf8"
        ) as out:
            for line in f:
                line = line.strip()
                for (rule_type, c_1, c_2), prob in rules.items():
                    if rule_type == "delete":
                        rand_delete = (
                            lambda c: "" if random.random() < prob and c == c_1 else c
                        )
                        line = "".join([rand_delete(c) for c in line])
                    elif rule_type == "replace":
                        rand_replace = (
                            lambda c: c_2 if random.random() < prob and c == c_1 else c
                        )
                        line = "".join([rand_replace(c) for c in line])
                    elif rule_type == "insert":
                        line = line + " "
                        rand_insert = (
                            lambda c: "{}{}".format(c_1, c)
                            if random.random() < prob
                            else c
                        )
                        line = "".join([rand_insert(c) for c in line])
                out.write(line.strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_src1",
        help="Source 1 from the training set. These are used to build the denoising rules.",
    )
    parser.add_argument(
        "--train_tgt",
        help="Manually corrected transcriptions from the training set. These are used to build the denoising rules.",
    )
    parser.add_argument(
        "--input",
        help="Input file to denoise. Typically these are the uncorrected src1 for pretraining.",
    )
    parser.add_argument("--output", help="Output filename.")
    args = parser.parse_args()

    denoiser = Denoiser()
    rules = denoiser.create_rules(src=args.train_src1, tgt=args.train_tgt)
    denoiser.denoise_file(rules=rules, input_file=args.input, output_file=args.output)
