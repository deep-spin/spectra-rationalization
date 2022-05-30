import torch


def remap_input_to_cf_vocab(input_ids, tokenizer, cf_tokenizer):
    """
    Remap an input from the original tokenizer (e.g. bert tokenizer) to
    another tokenizer (e.g. t5 tokenizer). For example,
    for the word 'CURIOUS', would get the following pieces:
        bert: ['C', '##UR', '##IO', '##US']
        t5  : ['▁C', 'URI', 'OUS']

    What we want is to map bert and t5 inputs s.t. len(bert_input) = len(t5_input).
    To achieve that, we tokenize each word individually and drop/repeat the
    position corresponding to the last piece. For the above example, we would
    get repeat_interleave counts such that:
        input:    ['C', '##UR', '##IO']
        cf_input: ['▁C', 'URI', 'OUS']

    If instead we originally got a tokenization like this:
        bert: ['C', '##URIOUS']
        t5  : ['▁C', 'URI', 'OUS']

    we would get repeat_interleave counts such that:
        input:    ['C', '##URIOUS', '##URIOUS']
        cf_input: ['▁C', 'URI', 'OUS']

    To achieve this, we just need the frequency that each token should be
    repeated in a interleaved manner (i.e. repeat_interleave counts).

    Args:
        input_ids: original input ids got from the factual tokenizer
        tokenizer: factual tokenizer
        cf_tokenizer: counterfactual tokenizer

    Returns:
        cf_input_counts: the frequency that each input_id will be repeated to
    """
    cf_input_counts = []
    for x_s in tokenizer.batch_decode(input_ids):
        x_counts_inner = []
        for word in x_s.split():
            # handle special tokens (CLS, SEP, PAD, UNK)
            if word in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, tokenizer.unk_token]:
                p_f = []
                p_cf = []
            else:
                a = 0 if tokenizer.cls_token is None and tokenizer.bos_token is None else 1
                b = None if tokenizer.eos_token is None and tokenizer.sep_token is None else -1
                p_f = tokenizer(word)['input_ids'][a:b]  # remove [cls] and [sep]
                a = 0 if cf_tokenizer.cls_token is None and cf_tokenizer.bos_token is None else 1
                b = None if cf_tokenizer.eos_token is None and cf_tokenizer.sep_token is None else -1
                p_cf = cf_tokenizer(word)['input_ids'][a:b]  # remove <s> and </s>
            # set c so that we repeat last piece
            if len(p_f) < len(p_cf):
                c = [1] * (len(p_f) - 1) + [1 + len(p_cf) - len(p_f)]
            # set c so that we drop the last pieces
            elif len(p_f) > len(p_cf):
                c = [1] * len(p_cf) + [0]*(len(p_f) - len(p_cf))
            # do nothing, they match sizes
            else:
                c = [1] * len(p_f)
            x_counts_inner.extend(c)
        cf_input_counts.append(torch.as_tensor(x_counts_inner))
    return cf_input_counts


def remap_input_to_cf_vocab_brute_force(bert_tokenizer, t5_tokenizer, x, z, mask):
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
