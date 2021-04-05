"""Script to compute character error rate and word error rate between a predicted text and its "gold" target text.

Usage:
python metrics.py --pred [predicted_filename] --tgt [target_filename]

Author: Shruti Rijhwani
Contact: srijhwan@cs.cmu.edu

Please cite:
OCR Post Correction for Endangered Language Texts (EMNLP 2020)
https://www.aclweb.org/anthology/2020.emnlp-main.478/
"""

import editdistance as ed
from collections import defaultdict
import argparse


class ErrorMetrics:
    def preprocess(self, text):
        preprocessed = " ".join(text.strip().split())
        return preprocessed

    def calculate_metrics(self, predicted_text, transcript):
        cer = ed.eval(predicted_text, transcript) / float(len(transcript))
        pred_spl = predicted_text.split()
        transcript_spl = transcript.split()
        wer = ed.eval(pred_spl, transcript_spl) / float(len(transcript_spl))
        return cer, wer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", help="Predicted text.")
    parser.add_argument("--tgt", help="Target text.")
    args = parser.parse_args()

    errors = defaultdict(lambda: 0)
    char_counts = defaultdict(lambda: 0)

    metrics = ErrorMetrics()

    pred_lines = open(args.pred, encoding="utf8").readlines()
    tgt_lines = open(args.tgt, encoding="utf8").readlines()

    predicted = []
    transcripts = []

    for pred_line, tgt_line in zip(pred_lines, tgt_lines):
        if not tgt_line.strip():
            continue

        predicted.append(metrics.preprocess(pred_line))
        transcripts.append(metrics.preprocess(tgt_line))

    cer, wer = metrics.calculate_metrics("\n".join(predicted), "\n".join(transcripts))

    print("CER: {}".format(cer * 100))
    print("WER: {}".format(wer * 100))
