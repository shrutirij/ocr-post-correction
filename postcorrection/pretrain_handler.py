"""[summary]

Author: Shruti Rijhwani
Contact: srijhwan@cs.cmu.edu

Please cite:
OCR Post Correction for Endangered Language Texts (EMNLP 2020)
https://www.aclweb.org/anthology/2020.emnlp-main.478/
"""

from lm_trainer import LMTrainer
from seq2seq_trainer import Seq2SeqTrainer
import _dynet as dy
from utils import DataReader
import logging


class PretrainHandler:
    def __init__(
        self,
        model,
        pretrain_src1,
        pretrain_src2,
        pretrain_tgt,
        pretrain_enc,
        pretrain_dec,
        pretrain_model,
        epochs,
    ):
        self.model = model
        self.lm_trainer = LMTrainer(model)
        self.seq2seq_trainer = Seq2SeqTrainer(model)

        if len(pretrain_src1) > 0 and pretrain_enc:
            logging.info("Pretraining src1 encoder")
            self.lm_trainer.train(
                pretrain_src1,
                self.model.src1_vocab,
                self.model.src1_lookup,
                self.model.enc1_fwd_lstm,
                self.model.enc1_bwd_lstm,
                self.model.pret1_w,
                self.model.pret1_b,
                attn=False,
                epochs=epochs,
            )

        if pretrain_src2 and len(pretrain_src2) > 0 and pretrain_enc:
            logging.info("Pretraining src2 encoder")
            self.lm_trainer.train(
                pretrain_src2,
                self.model.src2_vocab,
                self.model.src2_lookup,
                self.model.enc2_fwd_lstm,
                self.model.enc2_bwd_lstm,
                self.model.pret2_w,
                self.model.pret2_b,
                attn=False,
                epochs=epochs,
            )

        if len(pretrain_tgt) > 0 and pretrain_dec:
            logging.info("Pretraining tgt decoder")
            self.lm_trainer.train(
                pretrain_tgt,
                self.model.tgt_vocab,
                self.model.tgt_lookup,
                self.model.dec_lstm,
                None,
                self.model.dec_w,
                self.model.dec_b,
                attn=True,
                epochs=epochs,
            )

        if pretrain_model:
            logging.info("Pretraining seq2seq model")
            self.pretrain_model(pretrain_src1, pretrain_src2, pretrain_tgt, epochs)

        self.model.save()

    def pretrain_model(self, src1_path, src2_path, tgt_path, epochs):
        datareader = DataReader()
        data = datareader.read_parallel_data(self.model, src1_path, src2_path, tgt_path)
        self.seq2seq_trainer.train(
            train_data=data, val_data=[], epochs=epochs, pretrain=True,
        )
