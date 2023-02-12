import torch

from transformers import top_k_top_p_filtering


def get_t5_sentinel_ids(idx_a=32000, idx_b=32099):
    return list(range(idx_a, idx_b + 1))


def is_sentinel(x, idx_a=32000, idx_b=32099):
    return (x >= idx_a) & (x <= idx_b)


def fix_t5_generated_inputs(gen_ids, pad_id=0, idx_a=32000, idx_b=32099):
    """
    Fix T5 generated inputs by removing repeated sentinel tokens
    :param gen_ids: torch.LongTensor [B, T]
    :param pad_id: padding id
    :param idx_a: start of sentinel token range
    :param idx_b: end of sentinel token range
    """
    new_gen_ids = []
    for i in range(len(gen_ids)):
        is_sent = is_sentinel(gen_ids[i], idx_a, idx_b)
        is_fused = gen_ids[i] == gen_ids[i].roll(-1, dims=-1)
        valid_ids = gen_ids[i][~(is_sent & is_fused)]
        new_gen_ids.append(valid_ids)
    new_gen_ids = torch.nn.utils.rnn.pad_sequence(new_gen_ids, batch_first=True, padding_value=pad_id)
    new_gen_ids = new_gen_ids.to(gen_ids.device)
    return new_gen_ids


def get_mask_from_lengths(lengths):
    """
    Get a mask from lengths
    :param lengths: torch.LongTensor [B]
    :return: mask, torch.BoolTensor [B, T]
    """
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), -1) < lengths.unsqueeze(-1)
    return mask


def add_end_of_chunk_tokens_to_t5_input(x, z, mask, pad_id=0, eoc_id=32101, idx_a=32000, idx_b=32099):
    """
    Add end of chunk token to the end of each chunk

    :param x: input sequence, torch.FloatTensor [B, T]
    :param z: latent selection vector torch.FloarTensor [B, T]
    :param mask: mask for padding positions, torch.BoolTensor [B, T]
    :param pad_id: padding id
    :param eoc_id: end of chunk id
    :param idx_a: start of sentinel token range
    :param idx_b: end of sentinel token range
    :return: t5_x: new input ids for T5, torch.FloatTensor [B, T'],
             t5_z: new latent selection vector, torch.FloatTensor [B, T']
             t5_mask: new mask for padding positions, torch.BoolTensor [B, T']
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    x_is_sentinel = is_sentinel(x, idx_a, idx_b).long()
    max_num_sentinels = x_is_sentinel.sum(-1).max().item()

    x_eoc = torch.zeros(batch_size, max_num_sentinels, dtype=x.dtype, device=x.device) + eoc_id
    z_eoc = torch.ones(batch_size, max_num_sentinels, dtype=z.dtype, device=z.device)

    x_new = torch.cat([x, x_eoc], dim=-1)
    z_new = torch.cat([z, z_eoc], dim=-1)

    ordering_ori = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    ordering_eoc = torch.ones(batch_size, max_num_sentinels).long() * 999999
    for i in range(batch_size):
        nz = (is_sentinel(x[i], idx_a, idx_b)).long().nonzero()[:, 0]
        nz = nz[nz > 0]
        ordering_eoc[i, :len(nz)] = nz

    ordering = torch.cat([ordering_ori, ordering_eoc - 0.5], dim=-1)
    _, idxs = torch.sort(ordering, dim=-1, descending=False, stable=True)
    x_new = x_new.gather(1, idxs)
    z_new = z_new.gather(1, idxs)

    lengths = mask.long().sum(-1)
    lengths_new = lengths + x_is_sentinel.long().sum(-1)
    mask_new = get_mask_from_lengths(lengths_new)

    x_new = x_new.masked_fill(~mask_new, pad_id)
    z_new = z_new.masked_fill(~mask_new, 0)

    return x_new, z_new, mask_new


def make_input_for_ct5(x, e, z, mask, pad_id=0, idx_a=32000, idx_b=32099):
    t5_x, t5_e, t5_z, t5_mask = make_input_for_t5(x, e, z, mask, pad_id=pad_id, idx_a=idx_a, idx_b=idx_b)
    return add_end_of_chunk_tokens_to_t5_input(t5_x, t5_z, t5_mask, pad_id=pad_id, idx_a=idx_a, idx_b=idx_b)


def make_input_for_t5(x, e, z, mask, pad_id=0, idx_a=32000, idx_b=32099):
    """
    Replace masked positions by sentinel tokens

    :param x: input sequence, torch.FloatTensor [B, T]
    :param e: input embeddings, torch.FloatTensor [B, T, D]
    :param z: latent selection vector torch.FloarTensor [B, T]
    :param mask: mask for padding positions, torch.BoolTensor [B, T]
    :param pad_id: padding id
    :param idx_a: start of sentinel token range
    :param idx_b: end of sentinel token range
    :return: t5_x: new input ids for T5, torch.FloatTensor [B, T'],
             t5_e: new input embeddings for T5, torch.FloatTensor [B, T', D],
             t5_z: new latent selection vector, torch.FloatTensor [B, T']
             t5_mask: new mask for padding positions, torch.BoolTensor [B, T']
    """
    bs, seq_len, hdim = e.shape

    # for example:
    # z = [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]

    # leave only the first non-zero element in each contiguous chunk
    z_first = torch.relu(z - torch.cat((z.new_zeros(bs, 1), z[:, :-1]), dim=-1))
    # z_first = [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    # create a matrix of indices of shape (bs, seq_len)
    ar = torch.arange(seq_len, device=z.device).unsqueeze(0).expand(bs, -1).long()

    # select all tokens but the 2nd-n in each chunk
    z_all_but_succ = (1 - (z > 0).long() * (1 - (z_first > 0).long()))

    # get the ids of these tokens (and set 2nd-n tokens in each chunk to 999999)
    idxs_all_but_succ = z_all_but_succ * (ar + 1) + (1 - z_all_but_succ) * 999999

    # sort ids so that 999999 are at the end
    # (there are better ways to do this but sort is not very slow after all)
    idxs_all_but_succ = torch.sort(idxs_all_but_succ, stable=True, dim=-1).values

    # get the mask pointing to valid tokens
    z_mask = idxs_all_but_succ < 999999

    # discount 1 to get 0-index ids
    idxs_all_but_succ = idxs_all_but_succ - 1

    # for using gather, we need to replace these tokens by some valid index
    # so we use seq_len - 1 here, which will lead to pad tokens anyway
    idxs_all_but_succ = idxs_all_but_succ.masked_fill(~z_mask, seq_len - 1)

    # the new mask points to non padding tokens + eliminated tokens
    t5_mask = idxs_all_but_succ < mask.sum(-1).unsqueeze(-1)

    # gather the new input ids and fix padding positions
    t5_x = x.gather(1, idxs_all_but_succ)
    t5_x = t5_x.masked_fill(~t5_mask, pad_id)

    # the new z is simply z_first (the places where sentinels were inserted)
    # t5_z = z_first * t5_mask.float()
    t5_z = z_first.gather(1, idxs_all_but_succ)
    t5_z = t5_z.masked_fill(~t5_mask, 0)

    # expand as `e` so we can gather vectors from it
    t5_e = e.gather(1, idxs_all_but_succ.unsqueeze(-1).expand(-1, -1, hdim))

    # truncate to new max length
    max_len = t5_mask.sum(-1).max().item()
    t5_x = t5_x[:, :max_len]
    t5_e = t5_e[:, :max_len]
    t5_z = t5_z[:, :max_len]
    t5_mask = t5_mask[:, :max_len]

    # clamp valid input ids (might glue last generations as T5 has only 100 sentinels)
    t5_z_pos = (t5_z > 0).long()
    sentinel_ids = torch.clamp(idx_b + 1 - t5_z_pos.cumsum(dim=-1), min=idx_a, max=idx_b)
    # fix x by replacing selected tokens by sentinel ids
    t5_x = t5_z_pos * sentinel_ids + (1 - t5_z_pos) * t5_x

    return t5_x, t5_e, t5_z, t5_mask


def merge_inputs_for_t5(x, x_tilde, z, pad_id=0, eos_id=1, prefix_len=0, max_len=None, idx_a=32000, idx_b=32099):
    # get the counts needed to expand the input_ids into generated_ids
    gen_counts = get_new_frequencies_of_gen_ids_from_t5(
        x, x_tilde, pad_id=pad_id, eos_id=eos_id, idx_a=idx_a, idx_b=idx_b
    )
    # and vice-versa
    inp_counts = get_new_frequencies_of_gen_ids_from_t5(
        x_tilde, x, pad_id=pad_id, eos_id=eos_id, idx_a=idx_a, idx_b=idx_b
    )

    # expand x and z according to gen_counts
    x_rep = repeat_interleave_and_pad(x, gen_counts, pad_id=pad_id)
    z_tilde = repeat_interleave_and_pad(z, gen_counts, pad_id=pad_id)

    # expand x_tilde according to inp_counts
    x_tilde_rep = repeat_interleave_and_pad(x_tilde, inp_counts)

    # merge x_rep and x_tilde_rep into a single tensor
    x_tilde = merge_input_and_gen_ids(x_rep, x_tilde_rep, pad_id=pad_id, idx_a=idx_a, idx_b=idx_b)

    # fix the corner case of generating fewer tokens than what was selected
    original_seq_len = z_tilde.shape[-1]
    expanded_seq_len = x_tilde.shape[-1]
    if original_seq_len > expanded_seq_len:
        z_tilde = z_tilde[:, :expanded_seq_len]

    # remove labels in case they were prepended
    x_tilde = x_tilde[:, prefix_len:]
    z_tilde = z_tilde[:, prefix_len:]

    # if we generated too much, there isn't much we can do besides truncating
    x_tilde = x_tilde[:, :max_len]
    z_tilde = z_tilde[:, :max_len]

    return x_tilde, z_tilde


def get_new_frequencies_of_gen_ids_from_t5(x_ids, g_ids, pad_id=0, eos_id=1, idx_a=32000, idx_b=32099):
    """
    Get the number of tokens generated by T5 for each position of the input.

    :param x_ids: original input ids, torch.LongTensor of shape [B, T]
    :param g_ids: generated ids, torch.LongTensor of shape [B, T]
    :param pad_id: id of padding token
    :param eos_id: id of end-of-sequence token
    :param idx_a: id of first token of the new frequency range
    :param idx_b: id of last token of the new frequency range
    :return: new_freqs, torch.LongTensor of shape [B, T]
    """
    new_freqs = []
    for i in range(x_ids.shape[0]):
        # for example:
        # x = ['▁UN', '▁Chief', '▁says', '▁there', '▁is', '▁no', '▁way', '▁to', '<extra_id_0>', '▁in', '▁Syria', '</s>']
        # g = ['<extra_id_0>', '▁do', '▁so', '<extra_id_1>', '▁do', '▁so', '.', '</s>', '<pad>', '<pad>']

        # recover z from x_ids
        z_x = (x_ids[i] >= idx_a) & (x_ids[i] <= idx_b)          # select sentinel tokens
        z_x = z_x & (x_ids[i] != pad_id) & ~((x_ids[i] == eos_id) & (x_ids[i].roll(-1) == pad_id))  # no <pad> and </s>
        # for example:
        # z_x = tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

        # recover the complement of z from g_ids
        z_y = ~((g_ids[i] >= idx_a) & (g_ids[i] <= idx_b))       # select non sentinel tokens
        z_y = z_y & (g_ids[i] != pad_id) & ~((g_ids[i] == eos_id) & (g_ids[i].roll(-1) == pad_id))  # no <pad> and </s>
        # for example:
        # z_y = tensor([0, 1, 1, 0, 1, 1, 1, 0, 0, 0])

        # count the number of consecutive non-sentinel tokens
        outputs, counts = torch.unique_consecutive(z_y, return_counts=True)
        # for example:
        # outputs = tensor([0, 1, 0, 1, 0])
        # counts = tensor([1, 2, 1, 3, 3])

        # mask out invalid outputs due to end-of-sequence generation
        # (times 2 because sentinel tokens are also produced in the output)
        m = torch.arange(len(outputs), device=z_x.device) < (z_x.sum() * 2)
        # for example:
        # m = tensor([1, 1, 0, 0, 0])

        # count only the valid outputs
        outputs = outputs.masked_fill(~m, 0)
        counts = counts.masked_fill(~m, 1)
        # for example:
        # outputs = tensor([0, 1, 0, 0, 0])
        # counts = tensor([1, 2, 1, 1, 1])

        # convert counts to repeat_interleave frequencies
        n_x = 1 - z_x.clone().long()
        # for example:
        # n_x = tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1])

        # trick corner case:
        # the model generated fewer items than the number of sentinel tokens
        # in this case, we just count the first generated tokens
        if (n_x == 0).sum() > outputs.sum():
            # shell_logger.warning(
            #     'The number of generated chunks ({}) is less than the number of sentinel tokens ({}). '
            #     'Selecting only the first generated chunks.'.format(outputs.sum(), (n_x == 0).sum())
            # )
            m1 = n_x == 0
            m2 = m1.cumsum(0) <= outputs.sum()
            n_x[m1 & m2] = counts[outputs]
        else:
            n_x[n_x == 0] = counts[outputs]
        # n_x = tensor([1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1])

        # now we can repeat according to this new count, so we save them
        new_freqs.append(n_x)
    return new_freqs


def repeat_interleave_and_pad(x, counts, pad_id=0):
    """
    batch-wise repeat_interleave x according to counts,
    and then pad reminiscent positions with pad_id

    :param x: tensor with shape (batch_size, seq_len)
    :param counts: list of tensors with shape (seq_len,)
    :param pad_id: padding value
    """
    if counts[0] is None:
        return x
    seq_len = x.shape[-1]
    # if any(len(c) != seq_len for c in counts):
    #     print(list(map(len, counts)))  # why?
    x_new = [x[i].repeat_interleave(counts[i][:seq_len], dim=-1) for i in range(len(counts))]
    x_new = torch.nn.utils.rnn.pad_sequence(x_new, batch_first=True, padding_value=pad_id)
    return x_new.to(x.device)


def merge_input_and_gen_ids(input_ids_rep, generated_ids_rep, pad_id=0, idx_a=32000, idx_b=32099):
    """
    Merge the input ids and generated ids into one tensor.

    :param input_ids_rep: input ids after repeated_interleave, tensor of shape [B, T1]
    :param generated_ids_rep: generated ids after repeated_interleave, tensor of shape [B, T2]
    :param pad_id: id of padding token
    :param idx_a: id of first token of the new frequency range
    :param idx_b: id of last token of the new frequency range
    """
    # recover z from input ids and generated ids
    mask_inp = ~((input_ids_rep >= idx_a) & (input_ids_rep <= idx_b))
    mask_gen = ~((generated_ids_rep >= idx_a) & (generated_ids_rep <= idx_b))

    # get the length of the repeated tensors
    len_inp = input_ids_rep.shape[1]
    len_gen = generated_ids_rep.shape[1]

    # if we have less tokens in the original input, we truncate the generated ones
    if len_inp <= len_gen:
        m = mask_inp[:, :len_inp].long()
        merged_ids_rep = m * input_ids_rep[:, :len_inp] + (1 - m) * generated_ids_rep[:, :len_inp]

    # otherwise, we truncate the input ones
    else:
        m = mask_gen[:, :len_gen].long()
        merged_ids_rep = m * generated_ids_rep[:, :len_gen] + (1 - m) * input_ids_rep[:, :len_gen]

    # cap the new tensor to the new length since the merged_ids
    # can have more pad tokens than the necessary
    new_max_len = (merged_ids_rep != pad_id).sum(1).max().item()
    merged_ids_rep = merged_ids_rep[:, :new_max_len]

    return merged_ids_rep


def prepend_label_for_t5(
    x, y, z, mask, tokenizer, max_length=None, task_name='binary_classification', nb_classes=2
):
    """
    Prepend a label to the generated ids for T5 using the following format:
    `<LABEL>: <TOKENS>`

    :param y: labels, tensor of shape [B]
    :param x: input ids, tensor of shape [B, T]
    :param z: latent selections, tensor of shape [B, T]
    :param mask: input mask, tensor of shape [B, T]
    :param tokenizer: tokenizer instance
    :param max_length: maximum length of the input
    :param task_name: binary classification or nli
    :param nb_classes: number of classes
    :return:
        x_cf: tensors of shape [B, T + 2] with the prepended label template
        z_cf: tensors of shape [B, T + 2] with prepended `0s`
        mask_cf: tensors of shape [B, T + 2] with prepended `1s`
    """
    batch_size = y.shape[0]

    # manually create the ids of the template:
    if task_name == 'qe':
        prefix_ids_pos = torch.tensor(tokenizer.encode('ok:', add_special_tokens=False), device=y.device)
        prefix_ids_neg = torch.tensor(tokenizer.encode('bad:', add_special_tokens=False), device=y.device)
    else:
        prefix_ids_pos = torch.tensor(tokenizer.encode('positive:', add_special_tokens=False), device=y.device)
        prefix_ids_neg = torch.tensor(tokenizer.encode('negative:', add_special_tokens=False), device=y.device)
    prefix_ids_neu = torch.tensor(tokenizer.encode('neutral:', add_special_tokens=False), device=y.device)

    prefix_ids_pos = prefix_ids_pos[1:] if len(prefix_ids_pos) > 2 else prefix_ids_pos
    prefix_ids_neg = prefix_ids_neg[1:] if len(prefix_ids_neg) > 2 else prefix_ids_neg
    prefix_ids_neu = prefix_ids_neu[1:] if len(prefix_ids_neu) > 2 else prefix_ids_neu

    if task_name == 'binary_classification':
        prefix_ids = torch.where(
            y.unsqueeze(-1) == 0,
            prefix_ids_neg.unsqueeze(0).expand(batch_size, -1),
            prefix_ids_pos.unsqueeze(0).expand(batch_size, -1)
        )
    elif task_name == 'qe':
        prefix_ids = torch.where(
            y.unsqueeze(-1) == 0,
            prefix_ids_neg.unsqueeze(0).expand(batch_size, -1),  # bad has label=0
            prefix_ids_pos.unsqueeze(0).expand(batch_size, -1),  # ok has label=1
        )
    elif 'nli' in task_name:
        if nb_classes == 2:
            prefix_ids = torch.where(
                y.unsqueeze(-1) == 0,
                prefix_ids_pos.unsqueeze(0).expand(batch_size, -1),  # entailment has label=0
                prefix_ids_neg.unsqueeze(0).expand(batch_size, -1)   # contradiction has label=1
            )
        else:
            prefix_ids = torch.where(
                y.unsqueeze(-1) == 0,
                prefix_ids_pos.unsqueeze(0).expand(batch_size, -1),  # entailment has label=0
                torch.where(
                    y.unsqueeze(-1) == 1,
                    prefix_ids_neu.unsqueeze(0).expand(batch_size, -1),  # neutral has label=1
                    prefix_ids_neg.unsqueeze(0).expand(batch_size, -1)   # contradiction has label=2
                )
            )
    else:
        raise NotImplementedError

    # prepend prefix to the input (and cap if necessary)
    x_cf = torch.cat((prefix_ids, x), dim=1)
    x_cf = x_cf[:, :max_length] if max_length is not None else x_cf

    # and also deal with z and mask
    z_cf = None
    if z is not None:
        prefix_zeros = torch.zeros_like(prefix_ids)
        z_cf = torch.cat((prefix_zeros.to(z.dtype), z), dim=1)
        z_cf = z_cf[:, :max_length] if max_length is not None else z_cf

    mask_cf = None
    if mask is not None:
        prefix_ones = torch.ones_like(prefix_ids)
        mask_cf = torch.cat((prefix_ones.to(mask.dtype), mask), dim=1)
        mask_cf = mask_cf[:, :max_length] if max_length is not None else mask_cf

    return x_cf, z_cf, mask_cf


def prepend_label_for_t5_variable_length(
    x, y, z, mask, tokenizer, max_length=None, task_name='binary_classification', nb_classes=2, pad_id=0
):
    """
    Prepend a label to the generated ids for T5 using the following format:
    `<LABEL>: <TOKENS>`

    :param y: labels, tensor of shape [B]
    :param x: input ids, tensor of shape [B, T]
    :param z: latent selections, tensor of shape [B, T]
    :param mask: input mask, tensor of shape [B, T]
    :param tokenizer: tokenizer instance
    :param max_length: maximum length of the input
    :param task_name: binary classification or nli
    :param nb_classes: number of classes
    :param pad_id: id of the padding token

    :return:
        x_cf: tensors of shape [B, T + 2] with the prepended label template
        z_cf: tensors of shape [B, T + 2] with prepended `0s`
        mask_cf: tensors of shape [B, T + 2] with prepended `1s`
    """
    batch_size = y.shape[0]
    max_valid_len = mask.sum(dim=1).max().item()

    if task_name == 'binary_classification':
        prefix_ids_pos = torch.tensor(
            tokenizer.encode('label: positive input:', add_special_tokens=False),
            device=y.device
        )
        prefix_ids_neg = torch.tensor(
            tokenizer.encode('label: negative input:', add_special_tokens=False),
            device=y.device
        )
        prefix_ids = [
            prefix_ids_neg if y_ == 0 else prefix_ids_pos
            for y_ in y.tolist()
        ]
    elif task_name == 'qe':
        prefix_ids_ok = torch.tensor(
            tokenizer.encode('label: ok input:', add_special_tokens=False),
            device=y.device
        )
        prefix_ids_bad = torch.tensor(
            tokenizer.encode('label: bad input:', add_special_tokens=False),
            device=y.device
        )
        # bad has label=0
        # ok has label=1
        prefix_ids = [
            prefix_ids_bad if y_ == 0 else prefix_ids_ok
            for y_ in y.tolist()
        ]
    elif 'nli' in task_name:
        prefix_ids_ent = torch.tensor(
            tokenizer.encode('label: entailment input:', add_special_tokens=False),
            device=y.device
        )
        prefix_ids_neu = torch.tensor(
            tokenizer.encode('label: neutral input:', add_special_tokens=False),
            device=y.device
        )
        prefix_ids_con = torch.tensor(
            tokenizer.encode('label: contradiction input:', add_special_tokens=False),
            device=y.device
        )
        if nb_classes == 2:
            # entailment has label=0
            # contradiction has label=1
            prefix_ids = [
                prefix_ids_ent if y_ == 0 else prefix_ids_con
                for y_ in y.tolist()
            ]
        else:
            # entailment has label=0
            # neutral has label=1
            # contradiction has label=2
            prefix_ids = [
                prefix_ids_ent if y_ == 0 else
                prefix_ids_neu if y_ == 1 else
                prefix_ids_con
                for y_ in y.tolist()
            ]

    elif task_name == '20news':
        if nb_classes == 6:
            prefix_ids_prompt = [
                torch.tensor(tokenizer.encode('label: alt input:', add_special_tokens=False), device=y.device),
                torch.tensor(tokenizer.encode('label: comp input:', add_special_tokens=False), device=y.device),
                torch.tensor(tokenizer.encode('label: misc input:', add_special_tokens=False), device=y.device),
                torch.tensor(tokenizer.encode('label: rec input:', add_special_tokens=False), device=y.device),
                torch.tensor(tokenizer.encode('label: sci input:', add_special_tokens=False), device=y.device),
                torch.tensor(tokenizer.encode('label: talk input:', add_special_tokens=False), device=y.device),
            ]
        else:
            prefix_ids_prompt = [
                torch.tensor(tokenizer.encode('label: alt input:', add_special_tokens=False), device=y.device),
                torch.tensor(tokenizer.encode('label: comp input:', add_special_tokens=False), device=y.device),
                torch.tensor(tokenizer.encode('label: misc input:', add_special_tokens=False), device=y.device),
                torch.tensor(tokenizer.encode('label: rec input:', add_special_tokens=False), device=y.device),
                torch.tensor(tokenizer.encode('label: sci input:', add_special_tokens=False), device=y.device),
                torch.tensor(tokenizer.encode('label: soc input:', add_special_tokens=False), device=y.device),
                torch.tensor(tokenizer.encode('label: talk input:', add_special_tokens=False), device=y.device),
            ]

        prefix_ids = [prefix_ids_prompt[y_] for y_ in y.tolist()]
    else:
        raise NotImplementedError

    def concat_and_truncate(pre_ids, inp_ids, max_len):
        v = torch.nn.utils.rnn.pad_sequence(
            [torch.cat([pre_ids[i], inp_ids[i]]) for i in range(batch_size)],
            batch_first=True, padding_value=pad_id
        )
        max_len_cat = max(map(len, pre_ids)) + max_valid_len
        v = v[:, :max_len_cat]
        v = v[:, :max_len] if max_len is not None else v
        return v

    # prepend prefix to the input (and cap if necessary)
    x_cf = concat_and_truncate(prefix_ids, x, max_length)

    # and also deal with z and mask
    z_cf = None
    if z is not None:
        prefix_zeros = [torch.zeros_like(prefix_ids[i]).to(z.dtype) for i in range(batch_size)]
        z_cf = concat_and_truncate(prefix_zeros, z, max_length)

    mask_cf = None
    if mask is not None:
        prefix_ones = [torch.ones_like(prefix_ids[i]).to(mask.dtype) for i in range(batch_size)]
        mask_cf = concat_and_truncate(prefix_ones, mask, max_length)

    return x_cf, z_cf, mask_cf


def remove_label_for_t5_variable_length(
    x_cf, z_cf, mask_cf, max_length=None, task_name='binary_classification', pad_id=0
):
    """
    Remove the label from the generated ids for T5 using the following format:

    label: <LABEL> input: <TOKENS>

    :param x_cf: input ids, tensor of shape [B, T]
    :param z_cf: latent selections, tensor of shape [B, T]
    :param mask_cf: input mask, tensor of shape [B, T]
    :param max_length: maximum length of the input
    :param task_name: binary classification, nli, or qe
    :param pad_id: padding id

    :return:
        x_cf: tensors of shape [B, T'] without the prepended label template
        z_cf: tensors of shape [B, T'] without prepended `0s`
        mask_cf: tensors of shape [B, T'] without prepended `1s`
    """
    # 10 == ':' for regular tokenizer and 267 for multilingual tokenizer
    sel = x_cf == 10 if task_name != 'qe' else x_cf == 267
    sel[:, 1] = False  # ignore the first ':'
    sel_stop_idxs = sel.long().argmax(-1)

    def remove_and_truncate(v_cf, max_len, max_len_cat_gold=None):
        v = torch.nn.utils.rnn.pad_sequence(
            [v_cf[i, s + 1:] for i, s in enumerate(sel_stop_idxs.tolist())],
            batch_first=True, padding_value=pad_id
        )
        max_len_cat_ = (v != pad_id).long().sum(-1).max().item() if max_len_cat_gold is None else max_len_cat_gold
        v = v[:, :max_len_cat_]
        v = v[:, :max_len] if max_len is not None else v
        if max_len_cat_gold is None:
            return v, max_len_cat_
        return v

    x, max_len_cat = remove_and_truncate(x_cf, max_length)
    z = remove_and_truncate(z_cf, max_length, max_len_cat) if z_cf is not None else None
    mask = remove_and_truncate(mask_cf, max_length, max_len_cat) if mask_cf is not None else None
    return x, z, mask


def prepend_label_for_mice_t5(y_hat, x_cf, z_cf, mask_cf, tokenizer, task='imdb'):
    """
    Prepend a label to the generated ids for MICE T5 using the following format:

    `label: <LABEL>. input: <TOKENS>`

    where <LABEL> is either `negative` or `positive` and <TOKENS> is the input tokens.

    :param y_hat: classifier logits, tensor of shape [B, T, 2]
    :param x_cf: input ids, tensor of shape [B, T]
    :param z_cf: latent selections, tensor of shape [B, T]
    :param mask_cf: input mask, tensor of shape [B, T]
    :param tokenizer: tokenizer instance
    :param task: task name
    :return:
        x_cf: tensors of shape [B, T + 6] with the prepended label template
        z_cf: tensors of shape [B, T + 6] with prepended `1s`
        mask_cf: tensors of shape [B, T + 6] with prepended `1s`
    """
    if task == 'imdb':
        neg_id = tokenizer.vocab['▁negative']
        pos_id = tokenizer.vocab['▁positive']
        # manually create the ids of the template:
        # prefix_ids = tokenizer.tokenize('label: positive. input:')
        # prefix_ids = tokenizer.tokenize('label: negative. input:')
        prefix_ids = torch.tensor([3783, 10, -1, 5, 3785, 10]).to(x_cf.device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(x_cf.shape[0], -1)
        # replace -1 by neg or pos ids according to the factual prediction
        # but reverse labels to get counterfactuals
        y_hat_ids = torch.where(y_hat.argmax(-1) == 0, pos_id, neg_id)
        # merge prefix_ids and y_hat_ids
        y_hat_ids = y_hat_ids.unsqueeze(-1).repeat(1, prefix_ids.shape[-1])
        prefix_m = (prefix_ids == -1).long()
        prefix_ids = prefix_m * y_hat_ids + (1 - prefix_m) * prefix_ids
        # prepend prefix to the input
        x_cf = torch.cat((prefix_ids, x_cf), dim=1)
        # and also deal with z and mask
        prefix_zeros = torch.zeros_like(prefix_ids)
        prefix_ones = torch.ones_like(prefix_ids)
        z_cf = torch.cat((prefix_zeros, z_cf), dim=1)
        mask_cf = torch.cat((prefix_ones, mask_cf), dim=1)

    else:
        raise NotImplementedError

    return x_cf, z_cf, mask_cf


def sample_from_logits(logits, top_k=0, top_p=1.0, min_tokens_to_keep=1, num_samples=1):
    """
    Sample indices from the logits via top-k or nucleus sampling (top_p).

    :param logits: logits, tensor of shape [B, T, vocab_size]
    :param top_k: sample from the top k most likely tokens (if > 0)
    :param top_p: sample from the top p most likely tokens (with p = N * top_p)
    :param min_tokens_to_keep: minimum number of tokens to keep, even if we are sampling fewer than k tokens
    :param num_samples: number of samples to draw
    :return:
        sample: tensor of shape [B, T, num_samples]
    """
    filtered_logits = top_k_top_p_filtering(logits,
                                            top_k=top_k,
                                            top_p=top_p,
                                            min_tokens_to_keep=min_tokens_to_keep)
    filtered_probas = torch.softmax(filtered_logits, dim=-1)
    filtered_probas = filtered_probas.view(-1, filtered_probas.shape[-1])
    sample = torch.multinomial(filtered_probas, num_samples=num_samples)
    sample = sample.view(logits.shape[0], logits.shape[1], num_samples)
    return sample


def get_contrast_label(y: torch.Tensor, num_classes: int, task_name: str = "binary_classification"):
    """
    Get the contrast label for a given label y.

    :param y: label, tensor of shape [B]
    :param num_classes: number of classes
    :param task_name: task name
    :return:
        contrast_label: tensor of shape [B]
    """
    if task_name == "binary_classification" or task_name == "qe":
        contrast_label = 1 - y
    elif 'nli' in task_name:
        contrast_label = torch.randint(0, num_classes, y.shape).to(y.device)
        # entailment becomes contradiction
        contrast_label[y == 0] = 2
        # contradiction becomes entailment
        contrast_label[y == 2] = 0
        # for neutral we sample a new contrast label
        contrast_label[y == 1] = torch.where(
            torch.eq(contrast_label[y == 1], y[y == 1]),
            (contrast_label[y == 1] + 1) % num_classes,
            contrast_label[y == 1]
        )
    else:
        raise NotImplementedError
    return contrast_label
