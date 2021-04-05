"""Module for a multisource sequence-to-sequence model.

The model has either one or two encoders (corresponding to the number of input sources).
The attention mechanism concatenates the context vectors from both encoders, which is then used by the decoder to generate the output.

The model also has adaptations for the low-resource setting: copy mechanism, coverage mechanism, and diagonal attention loss.
These can be switched on or off as hyperparameters to the model.

The OCR post-correction training process uses early stopping on the validation set character error rate.

The model uses beam search for generation.

Copyright (c) 2021, Shruti Rijhwani
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


import _dynet as dy
from constants import (
    LSTM_NUM_OF_LAYERS,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    ATTENTION_SIZE,
    UNK,
    EOS,
    COV_LOSS_WEIGHT,
    DIAG_LOSS_WEIGHT,
)
from utils import DataReader, Hypothesis
import math
import numpy as np
import random
import logging


class TwoSourceModel:
    def __init__(
        self,
        src1_vocab,
        src2_vocab,
        tgt_vocab,
        single,
        pointer_gen,
        coverage,
        diag_loss,
        load_model,
        model_file,
        beam_size,
        best_val_cer,
    ):
        self.model = dy.ParameterCollection()

        self.src1_vocab = src1_vocab
        self.src2_vocab = src2_vocab
        self.tgt_vocab = tgt_vocab

        self.src1_lookup = self.model.add_lookup_parameters(
            (src1_vocab.length(), EMBEDDING_DIM)
        )
        self.src2_lookup = self.model.add_lookup_parameters(
            (src2_vocab.length(), EMBEDDING_DIM)
        )
        self.tgt_lookup = self.model.add_lookup_parameters(
            (tgt_vocab.length(), EMBEDDING_DIM)
        )

        self.enc1_fwd_lstm = dy.CoupledLSTMBuilder(
            LSTM_NUM_OF_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, self.model
        )
        self.enc1_bwd_lstm = dy.CoupledLSTMBuilder(
            LSTM_NUM_OF_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, self.model
        )
        self.pret1_w = self.model.add_parameters((src1_vocab.length(), HIDDEN_DIM))
        self.pret1_b = self.model.add_parameters((src1_vocab.length()))

        self.enc2_fwd_lstm = dy.CoupledLSTMBuilder(
            LSTM_NUM_OF_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, self.model
        )
        self.enc2_bwd_lstm = dy.CoupledLSTMBuilder(
            LSTM_NUM_OF_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, self.model
        )
        self.pret2_w = self.model.add_parameters((src2_vocab.length(), HIDDEN_DIM))
        self.pret2_b = self.model.add_parameters((src2_vocab.length()))

        self.att1_w1 = self.model.add_parameters((ATTENTION_SIZE, HIDDEN_DIM * 2))
        self.att1_w2 = self.model.add_parameters(
            (ATTENTION_SIZE, HIDDEN_DIM * LSTM_NUM_OF_LAYERS * 2)
        )
        self.att1_v = self.model.add_parameters((1, ATTENTION_SIZE))

        self.att2_w1 = self.model.add_parameters((ATTENTION_SIZE, HIDDEN_DIM * 2))
        self.att2_w2 = self.model.add_parameters(
            (ATTENTION_SIZE, HIDDEN_DIM * LSTM_NUM_OF_LAYERS * 2)
        )
        self.att2_v = self.model.add_parameters((1, ATTENTION_SIZE))

        self.dec_lstm = dy.CoupledLSTMBuilder(
            LSTM_NUM_OF_LAYERS, HIDDEN_DIM * 4 + EMBEDDING_DIM, HIDDEN_DIM, self.model
        )
        self.W_s = self.model.add_parameters((HIDDEN_DIM, HIDDEN_DIM * 4))
        self.b_s = self.model.add_parameters((HIDDEN_DIM))
        self.dec_w = self.model.add_parameters((tgt_vocab.length(), HIDDEN_DIM))
        self.dec_b = self.model.add_parameters((tgt_vocab.length()))

        # Pointer-generator parameters
        self.ptr_w_c = self.model.add_parameters((1, 2 * HIDDEN_DIM))
        self.ptr_w_s = self.model.add_parameters((1, 2 * HIDDEN_DIM))
        self.ptr_w_x = self.model.add_parameters((1, EMBEDDING_DIM + 4 * HIDDEN_DIM))

        # Coverage parameters
        self.w_cov = self.model.add_parameters((ATTENTION_SIZE, 1))

        self.single_source = single
        self.pointer_gen = pointer_gen
        self.coverage = coverage
        self.diag_loss = diag_loss
        self.model_file = model_file

        if load_model:
            self.model.populate(load_model)
            logging.info("Loaded model: {}".format(load_model))

        self.beam_size = beam_size
        self.best_val_cer = best_val_cer

    def save(self):
        self.model.save(self.model_file)

    def run_lstm(self, init_state, input_vecs):
        out_vectors = init_state.transduce(input_vecs)
        return out_vectors

    def embed_idx(self, idx_list, embed_lookup):
        return [embed_lookup[idx] for idx in idx_list]

    def encode(self, embeds, fwd_lstm, bwd_lstm):
        embeds_rev = list(reversed(embeds))
        fwd_vectors = self.run_lstm(fwd_lstm.initial_state(), embeds)
        bwd_vectors = self.run_lstm(bwd_lstm.initial_state(), embeds_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors

    def encoder_forward(self, src1, src2):
        embedded_src1 = self.embed_idx(src1, self.src1_lookup)
        if self.single_source:
            embedded_src2 = [dy.vecInput(EMBEDDING_DIM) for idx in src2]
        else:
            embedded_src2 = self.embed_idx(src2, self.src2_lookup)

        encoded_src1 = self.encode(
            embedded_src1, self.enc1_fwd_lstm, self.enc1_bwd_lstm
        )
        encoded_src2 = self.encode(
            embedded_src2, self.enc2_fwd_lstm, self.enc2_bwd_lstm
        )

        src1_mat = dy.concatenate_cols(encoded_src1)
        src1_w1dt = self.att1_w1 * src1_mat
        src2_mat = dy.concatenate_cols(encoded_src2)
        src2_w1dt = self.att2_w1 * src2_mat

        if not self.single_source:
            start = (
                self.W_s * dy.concatenate([encoded_src1[-1], encoded_src2[-1]])
                + self.b_s
            )
        else:
            start = (
                self.W_s
                * dy.concatenate([encoded_src1[-1], dy.vecInput(2 * HIDDEN_DIM)])
                + self.b_s
            )

        last_output_embeddings = self.tgt_lookup[self.tgt_vocab.str2int(EOS)]
        c1_t = dy.vecInput(2 * HIDDEN_DIM)
        c2_t = dy.vecInput(2 * HIDDEN_DIM)
        decoder_state = self.dec_lstm.initial_state([start, dy.tanh(start)]).add_input(
            dy.concatenate([c1_t, c2_t, last_output_embeddings])
        )
        return src1_mat, src2_mat, src1_w1dt, src2_w1dt, decoder_state

    def attend(self, input_mat, state, w1dt, w2, v, coverage):
        w2dt = w2 * dy.concatenate(list(state.s()))
        if coverage:
            w1dt = w1dt + self.w_cov * dy.transpose(coverage)
        a_t = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        a_t = dy.softmax(a_t)
        return a_t, (input_mat * a_t)

    def get_pointergen_probs(self, c_t, state, x_t, a_t, probs, src1):
        if not self.pointer_gen:
            return probs, 1.0
        unk_idx = self.tgt_vocab.str2int(UNK)
        p_gen = dy.logistic(
            self.ptr_w_c * c_t
            + self.ptr_w_s * dy.concatenate(list(state.s()))
            + self.ptr_w_x * x_t
        )
        gen_probs = probs * p_gen
        copy_probs = a_t * (1 - p_gen)
        copy_probs_update = []
        for i in gen_probs:
            copy_probs_update.append([i])
        for char, prob in zip(src1, copy_probs):
            cur_idx = self.tgt_vocab.str2int(self.src1_vocab.int2str(char))
            if cur_idx == unk_idx:
                continue
            if isinstance(cur_idx, int):
                copy_probs_update[cur_idx].append(prob)
            else:
                for idx in cur_idx:
                    copy_probs_update[idx].append(prob / len(cur_idx))
        sum_probs = dy.concatenate([dy.esum(exps) for exps in copy_probs_update])
        return sum_probs, p_gen.scalar_value()

    def get_coverage(self, a_t, prev_coverage, training=True):
        if not self.coverage:
            if not training:
                return None
            return dy.scalarInput(0), None
        coverage = a_t + prev_coverage
        if training:
            return (
                dy.sum_elems(dy.min_dim(dy.concatenate([a_t, coverage], d=1), d=1)),
                coverage,
            )
        return coverage

    def get_diag_loss(self, a_t, t):
        if self.diag_loss < 0:
            return dy.scalarInput(0)
        off_diag_elems = [dy.scalarInput(0)]
        for i, prob in enumerate(a_t):
            if i < (t - self.diag_loss) or i > (t + self.diag_loss):
                off_diag_elems.append(prob)
        return dy.esum(off_diag_elems)

    def decode_loss(self, src1, src2, tgt):
        src1_mat, src2_mat, src1_w1dt, src2_w1dt, decoder_state = self.encoder_forward(
            src1, src2
        )
        _, prev_coverage = self.get_coverage(
            a_t=dy.vecInput(len(src1)), prev_coverage=dy.vecInput(len(src1))
        )

        loss = []
        cov_loss = []
        diag_loss = []

        embedded_tgt = self.embed_idx(tgt, self.tgt_lookup)
        last_output_embeddings = self.tgt_lookup[self.tgt_vocab.str2int(EOS)]

        for t, (char, embedded_char) in enumerate(zip(tgt, embedded_tgt)):
            a_t, c1_t = self.attend(
                src1_mat,
                decoder_state,
                src1_w1dt,
                self.att1_w2,
                self.att1_v,
                prev_coverage,
            )
            if not self.single_source:
                _, c2_t = self.attend(
                    src2_mat, decoder_state, src2_w1dt, self.att2_w2, self.att2_v, None
                )
            else:
                c2_t = dy.vecInput(2 * HIDDEN_DIM)

            x_t = dy.concatenate([c1_t, c2_t, last_output_embeddings])
            decoder_state = decoder_state.add_input(x_t)

            out_vector = self.dec_w * decoder_state.output() + self.dec_b
            probs = dy.softmax(out_vector)
            probs, _ = self.get_pointergen_probs(
                c1_t, decoder_state, x_t, a_t, probs, src1
            )

            loss.append(-dy.log(dy.pick(probs, char)))
            cov_loss_cur, prev_coverage = self.get_coverage(a_t, prev_coverage)
            cov_loss.append(cov_loss_cur)
            diag_loss.append(self.get_diag_loss(a_t, t))

            last_output_embeddings = embedded_char

        loss = dy.esum(loss)
        cov_loss = dy.esum(cov_loss)
        diag_loss = dy.esum(diag_loss)
        return loss + COV_LOSS_WEIGHT * cov_loss + DIAG_LOSS_WEIGHT * diag_loss

    def get_loss(self, src1, src2, tgt):
        return self.decode_loss(src1, src2, tgt)

    def generate_beam(self, src1, src2):
        src1_mat, src2_mat, src1_w1dt, src2_w1dt, decoder_state = self.encoder_forward(
            src1, src2
        )

        hypothesis_list = [
            Hypothesis(
                text_list=[self.tgt_vocab.str2int(EOS)],
                decoder_state=decoder_state,
                c1_t=dy.vecInput(2 * HIDDEN_DIM),
                c2_t=dy.vecInput(2 * HIDDEN_DIM),
                prev_coverage=self.get_coverage(
                    a_t=dy.vecInput(len(src1)),
                    training=False,
                    prev_coverage=dy.vecInput(len(src1)),
                ),
                score=0.0,
                p_gens=[],
            )
        ]
        completed_list = []

        for t in range(int(len(src1) * 1.1)):
            new_hyp_list = []
            new_hyp_scores = []
            for hyp in hypothesis_list:
                last_output_embeddings = self.tgt_lookup[hyp.text_list[-1]]

                a_t, c1_t = self.attend(
                    src1_mat,
                    hyp.decoder_state,
                    src1_w1dt,
                    self.att1_w2,
                    self.att1_v,
                    hyp.prev_coverage,
                )
                if not self.single_source:
                    _, c2_t = self.attend(
                        src2_mat,
                        hyp.decoder_state,
                        src2_w1dt,
                        self.att2_w2,
                        self.att2_v,
                        None,
                    )
                else:
                    c2_t = dy.vecInput(2 * HIDDEN_DIM)

                x_t = dy.concatenate([c1_t, c2_t, last_output_embeddings])
                decoder_state = hyp.decoder_state.add_input(x_t)

                probs = dy.softmax(self.dec_w * decoder_state.output() + self.dec_b)
                probs, cur_p_gen = self.get_pointergen_probs(
                    c1_t, decoder_state, x_t, a_t, probs, src1
                )
                probs = probs.npvalue()

                for ind in range(len(probs)):
                    text_list = hyp.text_list + [ind]
                    p_gens = hyp.p_gens + [cur_p_gen]
                    score = (hyp.score + math.log(probs[ind])) / (len(text_list) ** 0.0)
                    coverage = self.get_coverage(a_t, hyp.prev_coverage, training=False)
                    new_hyp_list.append(
                        Hypothesis(
                            text_list=text_list,
                            decoder_state=decoder_state,
                            c1_t=c1_t,
                            c2_t=c2_t,
                            prev_coverage=coverage,
                            score=score,
                            p_gens=p_gens,
                        )
                    )
                    new_hyp_scores.append(score)

            top_inds = np.argpartition(np.array(new_hyp_scores), -self.beam_size)[
                -self.beam_size :
            ]
            new_hyp_list = np.array(new_hyp_list)[top_inds]

            hypothesis_list = []

            for new_hyp in new_hyp_list:
                if new_hyp.text_list[-1] == self.tgt_vocab.str2int(EOS) and t > 0:
                    completed_list.append(new_hyp)
                else:
                    hypothesis_list.append(new_hyp)

            if len(completed_list) >= self.beam_size:
                break

        if len(completed_list) == 0:
            sorted(hypothesis_list, key=lambda x: x.score, reverse=True)
            completed_list = [hypothesis_list[0]]

        for hyp in completed_list:
            hyp.text_list = [self.tgt_vocab.int2str(i) for i in hyp.text_list]

        top_hyp = sorted(completed_list, key=lambda x: x.score, reverse=True)[0]
        return "".join(top_hyp.text_list).replace(EOS, "").strip(), top_hyp.p_gens[1:-1]
