"""Script to create and save the character vocabulary for the models based on the input files.

The same vocabulary can be used for multiple models and experimental settings for a single document or language.

Copyright (c) 2021, Shruti Rijhwani
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
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
