import logging

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from rationalizers.explainers import available_explainers
from rationalizers.lightning_models.highlights.base import BaseRationalizer
from rationalizers.modules.scalar_mix import ScalarMixWithDropout
from rationalizers.modules.sentence_encoders import LSTMEncoder, MaskedAverageEncoder
from rationalizers.utils import get_z_stats, freeze_module, masked_average

shell_logger = logging.getLogger(__name__)


class CounterfactualRationalizer(BaseRationalizer):

    def __init__(
        self,
        tokenizer: object,
        nb_classes: int,
        is_multilabel: bool,
        h_params: dict,
    ):
        """
        :param tokenizer (object): torchnlp tokenizer object
        :param nb_classes (int): number of classes used to create the last layer
        :param multilabel (bool): whether the problem is multilabel or not (it depends on the dataset)
        :param h_params (dict): hyperparams dict. See docs for more info.
        """
        super().__init__(tokenizer, nb_classes, is_multilabel, h_params)

        ########################
        # hyperparams
        ########################
        # factual:
        self.gen_arch = h_params.get("gen_arch", "bert-base-multilingual-cased")
        self.pred_arch = h_params.get("pred_arch", "bert-base-multilingual-cased")
        self.gen_emb_requires_grad = h_params.get("gen_emb_requires_grad", False)
        self.pred_emb_requires_grad = h_params.get("pred_emb_requires_grad", False)
        self.gen_encoder_requires_grad = h_params.get("gen_encoder_requires_grad", True)
        self.pred_encoder_requires_grad = h_params.get("pred_encoder_requires_grad", True)
        self.shared_gen_pred = h_params.get("shared_gen_pred", False)
        self.use_scalar_mix = h_params.get("use_scalar_mix", True)
        self.dropout = h_params.get("dropout", 0.1)
        self.selection_space = h_params.get("selection_space", 'embedding')
        self.selection_vector = h_params.get("selection_vector", 'zero')
        self.selection_mask = h_params.get("selection_mask", True)
        self.selection_faithfulness = h_params.get("selection_faithfulness", True)
        # counterfactual:
        self.cf_gen_arch = h_params.get("cf_gen_arch", "bert-base-multilingual-cased")
        self.cf_pred_arch = h_params.get("cf_pred_arch", "bert-base-multilingual-cased")
        self.cf_gen_emb_requires_grad = h_params.get("cf_gen_emb_requires_grad", False)
        self.cf_pred_emb_requires_grad = h_params.get("cf_pred_emb_requires_grad", False)
        self.cf_gen_encoder_requires_grad = h_params.get("cf_gen_encoder_requires_grad", True)
        self.cf_pred_encoder_requires_grad = h_params.get("cf_pred_encoder_requires_grad", True)
        self.cf_shared_gen_pred = h_params.get("cf_shared_gen_pred", False)
        self.cf_use_scalar_mix = h_params.get("cf_use_scalar_mix", True)
        self.cf_dropout = h_params.get("cf_dropout", 0.1)
        self.cf_input_space = h_params.get("cf_input_space", 'ids')
        self.cf_selection_vector = h_params.get("cf_selection_vector", 'zero')
        self.cf_selection_mask = h_params.get("cf_selection_mask", True)
        self.cf_selection_faithfulness = h_params.get("cf_selection_faithfulness", True)
        self.cf_use_reinforce = h_params.get('cf_use_reinforce', True)
        self.cf_use_baseline = h_params.get('cf_use_baseline', True)
        # both:
        self.explainer_fn = h_params.get("explainer", True)
        self.explainer_pre_mlp = h_params.get("explainer_pre_mlp", True)
        self.temperature = h_params.get("temperature", 1.0)
        self.shared_preds = h_params.get("shared_preds", True)

        if self.shared_gen_pred:
            assert self.gen_arch == self.pred_arch
        if self.cf_shared_gen_pred:
            assert self.cf_gen_arch == self.cf_pred_arch

        ########################
        # factual flow
        ########################
        self.z = None
        self.mask_token_id = tokenizer.mask_token_id

        # generator module
        self.gen_hf = AutoModel.from_pretrained(self.gen_arch)
        self.gen_emb_layer = self.gen_hf.embeddings if 't5' not in self.gen_arch else self.gen_hf.shared
        self.gen_encoder = self.gen_hf.encoder
        self.gen_hidden_size = self.gen_hf.config.hidden_size
        self.gen_scalar_mix = ScalarMixWithDropout(
            mixture_size=self.gen_hf.config.num_hidden_layers+1,
            dropout=self.dropout,
            do_layer_norm=False,
        )

        # explainer
        explainer_cls = available_explainers[self.explainer_fn]
        self.explainer = explainer_cls(h_params, self.gen_hidden_size)
        self.explainer_mlp = nn.Sequential(
            nn.Linear(self.gen_hidden_size, self.gen_hidden_size),
            nn.Tanh(),
        )

        # predictor module
        if self.pred_arch == 'lstm':
            self.pred_encoder = LSTMEncoder(self.gen_hidden_size, self.gen_hidden_size, bidirectional=True)
            self.pred_hidden_size = self.gen_hidden_size * 2
        elif self.pred_arch == 'masked_average':
            self.pred_encoder = MaskedAverageEncoder()
            self.pred_hidden_size = self.gen_hidden_size
        else:
            self.pred_hf = self.gen_hf if self.shared_gen_pred else AutoModel.from_pretrained(self.pred_arch)
            self.pred_hidden_size = self.pred_hf.config.hidden_size
        self.pred_emb_layer = self.pred_hf.embeddings
        self.pred_encoder = self.pred_hf.encoder
        self.pred_scalar_mix = ScalarMixWithDropout(
            mixture_size=self.pred_hf.config.num_hidden_layers+1,
            dropout=self.dropout,
            do_layer_norm=False,
        )
        # predictor output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.pred_hidden_size, self.pred_hidden_size),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.pred_hidden_size, nb_classes),
            nn.Sigmoid() if not self.is_multilabel else nn.LogSoftmax(dim=-1),
        )

        ########################
        # counterfactual flow
        ########################
        # todo:
        # 1. (done) fix calls to cf_ objects
        # 2. (done) add hparams to cf flow
        # 3. (done) add option to use the same predictor for both flows
        # 4. (done) add option to use simpler predictors for both flows
        # 5. (done) implement repeat_interleave_as() function
        # 6. check how to use T5 as cf_generator:
        #   6.1. (done) may need to implement a remap vocab function
        #   6.2. (not needed) may need to subclass it to get the positions of predicted tokens
        # 7. (todo) implement a version of hf_counterfactual with reinforce
        # 8. (done) implement a version of hf_counterfactual with ST-gumbel-softmax
        self.z_bar = None

        # for reinforce
        self.log_prob_x_tilde = None
        self.n_points = 0
        self.mean_baseline = 0

        # counterfactual generator module
        self.cf_gen_hf = AutoModel.from_pretrained(self.cf_gen_arch)
        self.cf_gen_tokenizer = AutoTokenizer.from_pretrained(self.cf_gen_arch)
        self.cf_mask_token_id = self.cf_gen_tokenizer.mask_token_id
        self.cf_gen_emb_layer = self.cf_gen_hf.embeddings if 't5' not in self.cf_gen_arch else self.cf_gen_hf.shared
        self.cf_gen_encoder = self.cf_gen_hf.encoder
        self.cf_gen_lm_head = self.cf_gen_hf.lm_head
        self.cf_gen_hidden_size = self.cf_gen_hf.config.hidden_size
        self.cf_gen_scalar_mix = ScalarMixWithDropout(
            mixture_size=self.cf_gen_hf.config.num_hidden_layers+1,
            dropout=self.cf_dropout,
            do_layer_norm=False,
        )
        # counterfactual predictor module
        if self.cf_pred_arch == 'lstm':
            self.cf_pred_encoder = LSTMEncoder(self.cf_gen_hidden_size, self.cf_gen_hidden_size, bidirectional=True)
            self.cf_pred_hidden_size = self.cf_gen_hidden_size * 2
        elif self.cf_pred_hf == 'masked_average':
            self.cf_pred_encoder = MaskedAverageEncoder()
            self.cf_pred_hidden_size = self.cf_gen_hidden_size
        else:
            self.cf_pred_hf = self.cf_gen_hf if self.cf_shared_gen_pred else AutoModel.from_pretrained(self.cf_pred_arch)
            self.cf_pred_hidden_size = self.cf_pred_hf.config.hidden_size
        self.cf_pred_emb_layer = self.cf_pred_hf.embeddings
        self.cf_pred_encoder = self.cf_pred_hf.encoder
        self.cf_pred_scalar_mix = ScalarMixWithDropout(
            mixture_size=self.cf_pred_hf.config.num_hidden_layers+1,
            dropout=self.cf_dropout,
            do_layer_norm=False,
        )
        # counterfactual predictor output layer
        self.cf_output_layer = nn.Sequential(
            nn.Linear(self.cf_pred_hidden_size, self.cf_pred_hidden_size),
            nn.Tanh(),
            nn.Dropout(self.cf_dropout),
            nn.Linear(self.cf_pred_hidden_size, nb_classes),
            nn.Sigmoid() if not self.is_multilabel else nn.LogSoftmax(dim=-1),
        )

        ########################
        # weights details
        ########################
        # initialize params using xavier initialization for weights and zero for biases
        self.init_weights(self.explainer_mlp)
        self.init_weights(self.explainer)
        self.init_weights(self.output_layer)
        self.init_weights(self.cf_output_layer)

        # freeze embedding layers
        if not self.gen_emb_requires_grad:
            freeze_module(self.gen_emb_layer)
        if not self.pred_emb_requires_grad:
            freeze_module(self.pred_emb_layer)
        if not self.cf_gen_emb_requires_grad:
            freeze_module(self.cf_gen_emb_layer)
        if not self.cf_pred_emb_requires_grad:
            freeze_module(self.cf_pred_emb_layer)

        # freeze models
        if not self.gen_encoder_requires_grad:
            freeze_module(self.gen_encoder)
        if not self.pred_encoder_requires_grad:
            freeze_module(self.pred_encoder)
        if not self.cf_gen_encoder_requires_grad:
            freeze_module(self.cf_gen_encoder)
        if not self.cf_pred_encoder_requires_grad:
            freeze_module(self.cf_pred_encoder)

        # shared generator and predictor for factual flow
        if self.shared_gen_pred:
            del self.gen_scalar_mix  # unregister generator scalar mix
            self.gen_scalar_mix = self.pred_scalar_mix
        # shared generator and predictor for counterfactual flow
        if self.cf_shared_gen_pred:
            del self.cf_gen_scalar_mix  # unregister generator scalar mix
            self.cf_gen_scalar_mix = self.cf_pred_scalar_mix

        # shared factual and counterfactual predictors
        if self.shared_preds:
            del self.cf_pred_hf
            del self.cf_pred_emb_layer
            del self.cf_pred_encoder
            del self.cf_pred_hidden_size
            del self.cf_pred_scalar_mix
            del self.cf_output_layer
            self.cf_pred_hf = self.pred_hf
            self.cf_pred_emb_layer = self.pred_emb_layer
            self.cf_pred_encoder = self.pred_encoder
            self.cf_pred_hidden_size = self.pred_hidden_size
            self.cf_pred_scalar_mix = self.pred_scalar_mix
            self.cf_output_layer = self.output_layer

    def forward(
        self, x: torch.LongTensor, current_epoch=None, mask: torch.BoolTensor = None
    ):
        """
        Compute forward-pass.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: the output from SentimentPredictor. Torch.Tensor of shape [B, C]
        """
        z, y_hat = self.get_factual_flow(x, mask=mask)
        z_bar, y_bar_hat = self.get_counterfactual_flow(x, z, mask=mask)
        return (z, y_hat), (z_bar, y_bar_hat)

    def get_factual_flow(self, x, mask=None):
        # create mask for pretrained-LMs
        ext_mask = mask[:, None, None, :]  # add head and seq dimension
        ext_mask = ext_mask.to(dtype=self.dtype)  # fp16 compatibility
        ext_mask = (1.0 - ext_mask) * -10000.0  # will set softmax to zero

        gen_e = self.gen_emb_layer(x)
        if self.use_scalar_mix:
            gen_h = self.gen_encoder(gen_e, ext_mask, output_hidden_states=True).hidden_states
            gen_h = self.gen_scalar_mix(gen_h, mask)
        else:
            gen_h = self.gen_encoder(gen_e, ext_mask).last_hidden_state

        if self.explainer_pre_mlp:
            gen_h = self.explainer_mlp(gen_h)
        z = self.explainer(gen_h, mask)
        z_mask = (z * mask.float()).unsqueeze(-1)

        if self.selection_faithfulness is True:
            pred_e = self.pred_emb_layer(x)
        else:
            pred_e = gen_h

        if self.selection_vector == 'mask':
            # create an input with full mask tokens
            x_mask = torch.ones_like(x) * self.mask_token_id
            pred_e_mask = self.pred_emb_layer(x_mask)
        else:
            pred_e_mask = torch.zeros_like(pred_e)

        if self.selection_space == 'token':
            z_mask_bin = (z_mask > 0).float()
            pred_e = pred_e * z_mask_bin + pred_e_mask * (1 - z_mask_bin)
        elif self.selection_space == 'embedding':
            pred_e = pred_e * z_mask + pred_e_mask * (1 - z_mask)

        if self.selection_mask:
            ext_mask *= (z_mask.squeeze(-1)[:, None, None, :] > 0.0)

        if self.use_scalar_mix:
            pred_h = self.pred_encoder(pred_e, ext_mask, output_hidden_states=True).hidden_states
            pred_h = self.pred_scalar_mix(pred_h, mask)
        else:
            pred_h = self.pred_encoder(pred_e, ext_mask).last_hidden_state

        summary = masked_average(pred_h, mask)
        y_hat = self.output_layer(summary)

        return z, y_hat

    def get_counterfactual_flow(self, x, z, mask=None):
        x, z, mask = self.remap_input_to_counterfactual_vocab(
            self.tokenizer, self.cf_gen_tokenizer, x, z, mask
        )

        # prepare input for the generator LM
        e = self.cf_gen_emb_layer(x) if self.cf_input_space == 'ids' else x
        e_mask = self.cf_gen_emb_layer(torch.ones_like(e) * self.cf_mask_token_id)

        # fix inputs for t5
        if 't5' in self.cf_gen_arch:
            e, z, mask = make_input_for_t5(e, z, mask)
            # create sentinel tokens
            e_mask = self.cf_gen_emb_layer(z.cumsum(dim=-1) - 1 + 32000)

        # create mask for pretrained-LMs
        ext_mask = (1.0 - mask[:, None, None, :].to(self.dtype)) * -10000.0

        # get the complement of z
        z_bar = 1 - z
        z_bar = (z_bar * mask.float()).unsqueeze(-1)

        # get mask for the generator LM
        gen_ext_mask = ext_mask * (z_bar.squeeze(-1)[:, None, None, :] > 0.0).float()

        # diff where
        s_bar = e * z_bar + e_mask * (1 - z_bar)

        # pass (1-z)-masked inputs
        if self.cf_use_scalar_mix:
            cf_gen_enc_out = self.cf_gen_encoder(s_bar, gen_ext_mask, output_hidden_states=True)
            h_tilde = cf_gen_enc_out.hidden_states
            h_tilde = self.cf_gen_scalar_mix(h_tilde, mask)
        else:
            cf_gen_enc_out = self.cf_gen_encoder(s_bar, gen_ext_mask)
            h_tilde = cf_gen_enc_out.last_hidden_state

        # sample from LM
        if self.cf_use_reinforce:
            if 't5' in self.cf_gen_arch:
                # fix last hidden state as the output of scalar mix
                cf_gen_enc_out.last_hidden_state = h_tilde
                # sample autoregressively
                gen_ids, logits = self.cf_gen_hf.generate(
                    encoder_outputs=cf_gen_enc_out, attention_mask=mask,
                    output_scores=True, **self.cf_sample_kwargs
                )
            else:
                # sample directly from the output layer
                logits = self.cf_gen_lm_head(h_tilde)
                gen_ids = logits.argmax(dim=-1)
        else:
            # use the ST-gumbel-softmax trick
            logits = self.cf_gen_lm_head(h_tilde)
            gen_ids = nn.functional.gumbel_softmax(logits, hard=True, dim=- 1)

        # compute log proba
        self.log_prob_x_tilde = torch.log_softmax(logits, dim=-1)

        # expand z to account for the new tokens
        x_tilde = gen_ids
        z_tilde = repeat_interleave_as_gen_ids(z, gen_ids)
        mask_tilde = repeat_interleave_as_gen_ids(mask, gen_ids)
        ext_mask_tilde = (1.0 - mask_tilde[:, None, None, :].to(self.dtype)) * -10000.0

        if self.cf_selection_faithfulness is True:
            # get the predictor embeddings corresponding to x_tilde
            pred_e = self.cf_pred_emb_layer(x_tilde)
        else:
            # get the generator hidden states
            pred_e = h_tilde

        # get the replacement vector for non-selected positions
        pred_e_mask = torch.zeros_like(pred_e)
        if self.cf_selection_vector == 'mask':
            pred_e_mask = self.cf_pred_emb_layer(torch.ones_like(x_tilde) * self.mask_token_id)

        # input selection
        s_tilde = pred_e * z_tilde + pred_e_mask * (1 - z_tilde)

        # whether we want to mask out non-selected elements during self-attention
        if self.cf_selection_mask:
            ext_mask_tilde *= (z_tilde.squeeze(-1)[:, None, None, :] > 0.0).float()

        # pass the selected inputs through the encoder
        if self.cf_use_scalar_mix:
            pred_h = self.cf_pred_encoder(s_tilde, ext_mask_tilde, output_hidden_states=True).hidden_states
            pred_h = self.cf_pred_scalar_mix(pred_h, mask_tilde)
        else:
            pred_h = self.cf_pred_encoder(s_tilde, ext_mask_tilde).last_hidden_state

        # get predictions
        summary = masked_average(pred_h, mask_tilde)
        y_hat = self.cf_output_layer(summary)

        return z_bar.squeeze(-1), y_hat

    def get_loss(self, y_hat, y, prefix, mask=None):
        pass

    def get_factual_loss(self, y_hat, y, prefix, mask=None):
        """
        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        stats = {}
        loss_vec = self.criterion(y_hat, y)  # [B] or [B,C]
        # main MSE loss for p(y | x,z)
        if not self.is_multilabel:
            loss = loss_vec.mean(0)  # [B,C] -> [B]
            stats["mse"] = loss.item()
        else:
            loss = loss_vec.mean()  # [1]
            stats["criterion"] = loss.item()  # [1]

        # latent selection stats
        num_0, num_c, num_1, total = get_z_stats(self.explainer.z, mask)
        stats[prefix + "_p0"] = num_0 / float(total)
        stats[prefix + "_pc"] = num_c / float(total)
        stats[prefix + "_p1"] = num_1 / float(total)
        stats[prefix + "_ps"] = (num_c + num_1) / float(total)

        return loss, stats

    def get_cunterfactual_loss(self, y_hat, y, prefix, mask=None, baseline=False):
        """
        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        stats = {}
        loss_vec = self.criterion(y_hat, y)  # [B] or [B,C]

        # x = h
        # y = y_tilde
        # z = x_tilde

        # main MSE loss for p(y | x, z)
        if not self.is_multilabel:
            loss_vec = loss_vec.mean(1)  # [B,C] -> [B]

        loss = loss_vec.mean()  # [1]
        if not self.is_multilabel:
            stats["mse"] = loss.item()  # [1]

        # recover z
        z = (1 - self.z_bar.squeeze(-1)) * mask.float()  # [B, T]

        # get P(z = 0 | x) and P(z = 1 | x)
        logp_z0 = 1 - self.log_prob_x_tilde  # [B,T], log P(z = 0 | x)
        logp_z1 = self.log_prob_x_tilde  # [B,T], log P(z = 1 | x)

        # compute log p(z|x) for each case (z==0 and z==1) and mask
        logpz = torch.where(z == 0, logp_z0, logp_z1)
        logpz = torch.where(mask, logpz, logpz.new_zeros([1]))

        # compute generator loss
        cost_vec = loss_vec.detach()
        # cost_vec is neg reward
        cost_logpz = ((cost_vec - self.mean_baseline) * logpz.sum(1)).mean(0)

        # MSE with regularizers = neg reward
        obj = cost_vec.mean()
        stats["obj"] = obj.item()

        # add baseline
        if self.cf_use_baseline:
            self.n_points += 1.0
            self.mean_baseline += (cost_vec.detach().mean() - self.mean_baseline) / self.n_points

        # pred diff doesn't do anything if only 1 aspect being trained
        if not self.is_multilabel:
            pred_diff = y_hat.max(dim=1)[0] - y_hat.min(dim=1)[0]
            pred_diff = pred_diff.mean()
            stats["pred_diff"] = pred_diff.item()

        # generator cost
        stats["cost_g"] = cost_logpz.item()

        # predictor cost
        stats["cost_p"] = loss.item()

        # latent selection stats
        num_0, num_c, num_1, total = get_z_stats(z, mask)
        stats["p0"] = num_0 / float(total)
        stats["pc"] = num_c / float(total)
        stats["p1"] = num_1 / float(total)
        stats["selected"] = num_1
        stats["total"] = float(total)

        main_loss = loss + cost_logpz
        stats["main_loss"] = main_loss.item()
        return main_loss, stats


def make_input_for_t5(e, z, mask):
    bs, seq_len = z.shape
    ar = torch.arange(seq_len).unsqueeze(0).expand(bs, -1)
    z_first = z - torch.cat((z.new_zeros(bs, 1), z[:, :-1]), dim=-1)
    z_all_but_succ = (1 - z * (1 - (z_first > 0).long())) * (ar + 1)
    z_all_but_succ = z_all_but_succ.masked_fill(z_all_but_succ == 0, seq_len + 1) - 1
    z_all_but_succ = torch.sort(z_all_but_succ, stable=True, dim=-1).values
    z_mask = z_all_but_succ < seq_len
    z_all_but_succ = z_all_but_succ.masked_fill(~z_mask, seq_len - 1)
    return e.gather(1, z_all_but_succ), z_first * z_mask.float(), mask & z_mask


def repeat_interleave_as_gen_ids(x_ids, g_ids, pad_id=0, idx_a=32000, idx_b=32099):
    x_rep = []
    for i in range(x_ids.shape[0]):
        z_x = (x_ids[i] >= idx_a) & (x_ids[i] <= idx_b)  # select sentinel tokens
        z_x = z_x & (x_ids[i] != pad_id) & (x_ids[i] != 1)    # ignore <pad> and </s>
        z_y = ~((g_ids[i, 1:] >= idx_a) & (g_ids[i, 1:] <= idx_b))  # select non sentinel tokens
        z_y = z_y & (g_ids[i, 1:] != pad_id) & (g_ids[i, 1:] != 1)       # ignore <pad> and </s>
        # count the number of consecutive non-sentinel tokens
        outputs, counts = torch.unique_consecutive(z_y, return_counts=True)
        # mask out invalid outputs due to end-of-sequence generation
        m = torch.arange(len(outputs), device=z_x.device) < (z_x.sum() * 2)
        outputs = outputs.masked_fill(~m, 0)
        counts = counts.masked_fill(~m, 1)
        # convert counts to repeat_interleave frequencies
        n_x = 1 - z_x.clone().long()
        n_x[n_x == 0] = counts[outputs]
        x_rep.append(torch.repeat_interleave(x_ids[i], n_x, dim=-1))
    return torch.nn.utils.rnn.pad_sequence(x_rep, batch_first=True, padding_value=pad_id)


@torch.no_grad()
def remap_input_to_counterfactual_vocab(bert_tokenizer, t5_tokenizer, x, z, mask):
    x_str = bert_tokenizer.batch_decode(x)
    x_new = []
    x_counts = []
    for x_s in x_str:
        x_new_inner = []
        x_counts_inner = []
        for word in x_s.split():
            if word == bert_tokenizer.cls_token:
                p_bert = [bert_tokenizer.cls_token_id]
                p_t5 = [t5_tokenizer.vocab['X']]
            elif word == bert_tokenizer.sep_token:
                p_bert = [bert_tokenizer.sep_token_id]
                p_t5 = [t5_tokenizer.eos_token_id]
            elif word == bert_tokenizer.pad_token:
                p_bert = [bert_tokenizer.pad_token_id]
                p_t5 = [t5_tokenizer.pad_token_id]
            elif word == bert_tokenizer.unk_token:
                p_bert = [bert_tokenizer.unk_token_id]
                p_t5 = [t5_tokenizer.unk_token_id]
            else:
                p_bert = bert_tokenizer(word)['input_ids'][1:-1]  # remove [cls] and [sep]
                p_t5 = t5_tokenizer(word)['input_ids'][:-1]  # remove </s>
            if len(p_bert) < len(p_t5):
                c = [1] * (len(p_bert) - 1) + [1 + len(p_t5) - len(p_bert)]
            elif len(p_bert) > len(p_t5):
                c = [1] * len(p_t5) + [0]*(len(p_bert) - len(p_t5))
            else:
                c = [1] * len(p_bert)
            x_counts_inner.extend(c)
            x_new_inner.extend(p_t5)
        x_counts.append(torch.as_tensor(x_counts_inner))
        x_new.append(torch.as_tensor(x_new_inner))

    z_new = [z[i].repeat_interleave(x_counts[i], dim=-1) for i in range(len(x_str))]
    mask_new = [mask[i].repeat_interleave(x_counts[i], dim=-1) for i in range(len(x_str))]

    x_new_pt = torch.nn.utils.rnn.pad_sequence(x_new, batch_first=True, padding_value=t5_tokenizer.pad_token_id)
    z_new_pt = torch.nn.utils.rnn.pad_sequence(z_new, batch_first=True, padding_value=0)
    mask_new_pt = torch.nn.utils.rnn.pad_sequence(mask_new, batch_first=True, padding_value=0)
    return x_new_pt, z_new_pt, mask_new_pt.bool()


@torch.no_grad()
def remap_input_to_counterfactual_vocab_brute_force(bert_tokenizer, t5_tokenizer, x, z, mask):
    x_new = []
    x_counts = []
    for x_i in x:
        x_new_inner = []
        x_counts_inner = []
        for word in bert_tokenizer.convert_ids_to_tokens(x_i):
            word = word.replace('##', '')
            if word == bert_tokenizer.cls_token:
                p_bert = [bert_tokenizer.cls_token_id]
                p_t5 = [t5_tokenizer.vocab['X']]
            elif word == bert_tokenizer.sep_token:
                p_bert = [bert_tokenizer.sep_token_id]
                p_t5 = [t5_tokenizer.eos_token_id]
            elif word == bert_tokenizer.pad_token:
                p_bert = [bert_tokenizer.pad_token_id]
                p_t5 = [t5_tokenizer.pad_token_id]
            elif word == bert_tokenizer.unk_token:
                p_bert = [bert_tokenizer.unk_token_id]
                p_t5 = [t5_tokenizer.unk_token_id]
            else:
                p_bert = bert_tokenizer(word)['input_ids'][1:-1]  # remove [cls] and [sep]
                p_t5 = t5_tokenizer(word)['input_ids'][:-1]  # remove </s>
            if len(p_bert) < len(p_t5):
                c = [1] * (len(p_bert) - 1) + [1 + len(p_t5) - len(p_bert)]
            elif len(p_bert) > len(p_t5):
                c = [1] * len(p_t5) + [0]*(len(p_bert) - len(p_t5))
            else:
                c = [1] * len(p_bert)
            x_counts_inner.extend(c)
            x_new_inner.extend(p_t5)
        x_counts.append(torch.as_tensor(x_counts_inner))
        x_new.append(torch.as_tensor(x_new_inner))
    z_new = [z[i].repeat_interleave(x_counts[i], dim=-1) for i in range(len(x))]
    mask_new = [mask[i].repeat_interleave(x_counts[i], dim=-1) for i in range(len(x))]
    x_new_pt = torch.nn.utils.rnn.pad_sequence(x_new, batch_first=True, padding_value=t5_tokenizer.pad_token_id)
    z_new_pt = torch.nn.utils.rnn.pad_sequence(z_new, batch_first=True, padding_value=0)
    mask_new_pt = torch.nn.utils.rnn.pad_sequence(mask_new, batch_first=True, padding_value=0)
    return x_new_pt, z_new_pt, mask_new_pt.bool()
