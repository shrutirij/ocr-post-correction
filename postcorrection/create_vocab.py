"""Script to create and save the character vocabulary for the models based on the input files.

The same vocabulary can be used for multiple models and experimental settings for a single document or language.

Author: Shruti Rijhwani
Contact: srijhwan@cs.cmu.edu

Please cite:
OCR Post Correction for Endangered Language Texts (EMNLP 2020)
https://www.aclweb.org/anthology/2020.emnlp-main.478/
"""

from utils import CharVocab
import argparse
import json


def create_vocab(filepaths):
    return CharVocab(filepaths)


def save_vocab(vocab, output_folder, filename):
    with open("{}/{}.json".format(output_folder, filename), "w", encoding="utf8") as f:
        json.dump(vocab.get_lookup(), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src1_files", nargs="+", default=[])
    parser.add_argument("--src2_files", nargs="+", default=[])
    parser.add_argument("--tgt_files", nargs="+", default=[])
    parser.add_argument("--output_folder")
    args = parser.parse_args()

    all_paths = [args.src1_files, args.src2_files, args.tgt_files]
    output_names = ["src1", "src2", "tgt"]

    for filepaths, output_name in zip(all_paths, output_names):
        vocab = create_vocab(filepaths)
        save_vocab(vocab, args.output_folder, output_name)
