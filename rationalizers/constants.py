# lowercased special tokens
PAD = "<pad>"
UNK = "<unk>"
EOS = "</s>"
SOS = "<s>"
SEP = "<SEP>"


# special tokens id
PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
SOS_ID = 3
SEP_ID = 4


def update_constants(tokenizer):
    # update constants
    global PAD_ID, UNK_ID, EOS_ID, SOS_ID, SEP_ID, PAD, UNK, EOS, SOS, SEP
    PAD_ID = tokenizer.pad_token_id
    UNK_ID = tokenizer.unk_token_id
    EOS_ID = tokenizer.eos_token_id
    SOS_ID = tokenizer.bos_token_id
    SEP_ID = tokenizer.sep_token_id
    PAD = tokenizer.pad_token
    UNK = tokenizer.unk_token
    EOS = tokenizer.eos_token
    SOS = tokenizer.bos_token
    SEP = tokenizer.sep_token
