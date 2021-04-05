"""Module to train an LSTM with a language model objective.

This loss function is used to pretrain the encoder and decoder LSTMs before training the post-correction model.

Copyright (c) 2021, Shruti Rijhwani
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


import dynet as dy
from constants import EOS, HIDDEN_DIM
import logging
import random
from utils import DataReader


class LMTrainer:
    def __init__(self, model):
        self.model = model

    def lm_loss(self, chars, lookup, lstm, w, b, attn):
        last_output_embeddings = lookup[chars[0]]
        lm_state = lstm.initial_state()
        losses = []

        for char in chars[1:]:
            if attn:
                last_output_embeddings = dy.concatenate(
                    [dy.vecInput(4 * HIDDEN_DIM), last_output_embeddings]
                )
            lm_state = lm_state.add_input(last_output_embeddings)
            out_vector = w * lm_state.output() + b
            losses.append(dy.pickneglogsoftmax(out_vector, char))
            last_output_embeddings = lookup[char]
        return dy.esum(losses)

    def train(
        self, filepath, vocab, embed_lookup, fwd_lstm, bwd_lstm, w, b, attn, epochs
    ):
        trainer = dy.AdamTrainer(self.model.model)
        data_reader = DataReader()
        data = data_reader.read_single_source_data(filepath, vocab)

        for e in range(epochs):
            logging.info("Pretrain epoch: %d" % e)
            epoch_loss = 0.0
            random.shuffle(data)

            for i in range(0, len(data), 32):
                cur_size = min(32, len(data) - i)
                losses_fwd = []
                losses_bwd = []
                dy.renew_cg()
                for chars in data[i : i + cur_size]:
                    losses_fwd.append(
                        self.lm_loss(chars, embed_lookup, fwd_lstm, w, b, attn)
                    )
                    if bwd_lstm:
                        losses_bwd.append(
                            self.lm_loss(
                                list(reversed(chars)),
                                embed_lookup,
                                bwd_lstm,
                                w,
                                b,
                                attn,
                            )
                        )
                if bwd_lstm:
                    batch_loss = (
                        dy.esum(losses_fwd) / cur_size + dy.esum(losses_bwd) / cur_size
                    )
                else:
                    batch_loss = dy.esum(losses_fwd) / cur_size
                batch_loss.backward()
                trainer.update()
                epoch_loss += batch_loss.scalar_value()
            logging.info("Pretrain epoch loss: %0.4f" % (epoch_loss / len(data)))
