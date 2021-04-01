"""[summary]

Author: Shruti Rijhwani
Contact: srijhwan@cs.cmu.edu

Please cite:
OCR Post Correction for Endangered Language Texts (EMNLP 2020)
https://www.aclweb.org/anthology/2020.emnlp-main.478/
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
    parser.add_argument("--train_src1")
    parser.add_argument("--train_tgt")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    denoiser = Denoiser()
    rules = denoiser.create_rules(src=args.train_src1, tgt=args.train_tgt)
    denoiser.denoise_file(rules=rules, input_file=args.input, output_file=args.output)
