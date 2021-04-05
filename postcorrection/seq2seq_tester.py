"""Module with functions for using a trained post-correction model to produce predictions on unseen input data.

The module can be used with a test set in order to get a text output and CER/WER metrics (lines 26). 
It can also be used without a target prediction to only get the predicted output (line 40).

Copyright (c) 2021, Shruti Rijhwani
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


import dynet as dy
from utils import DataReader, ErrorMetrics


class Seq2SeqTester:
    def __init__(self, model, output_name):
        self.model = model
        self.datareader = DataReader()
        self.metrics = ErrorMetrics()
        self.output_name = output_name

    def test(self, src1, src2, tgt):
        if tgt:
            data = self.datareader.read_parallel_data(self.model, src1, src2, tgt)
            output_name = "{}_{}".format(self.output_name, src1.split("/")[-1])
            cer, wer = self.metrics.get_average_cer(
                self.model,
                data,
                output_file=open(
                    "{}.output".format(output_name), "w", encoding="utf-8"
                ),
                write_pgens=False,
            )
            with open("{}.metrics".format(output_name), "w") as output_file:
                output_file.write("TEST CER: %0.4f\n" % (cer))
                output_file.write("TEST WER: %0.4f\n" % (wer))
        else:
            output_file = open(
                "{}_{}.output".format(self.output_name, src1.split("/")[-1]),
                "w",
                encoding="utf8",
            )
            data = self.datareader.read_test_data(self.model, src1, src2)
            for src1, src2 in data:
                if len(src1) == 0 or len(src2) == 0:
                    output_file.write("\n")
                    continue
                dy.renew_cg()
                output, _ = self.model.generate_beam(src1, src2)
                output_file.write(str(output) + "\n")
            output_file.close()
