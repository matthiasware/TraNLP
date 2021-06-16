def get_attn_pad_mask(x):
    # x (b, l)
    batch_size, d_sentence = x.size()
    pad_attn_mask = x.data.eq(0).unsqueeze(1)

    # (b, l, l)
    pad_attn_mask = pad_attn_mask.expand(batch_size, d_sentence, d_sentence)
    return pad_attn_mask


def get_attn_mask(x, n_heads=None):
    mask = get_attn_pad_mask(x)
    if n_heads is not None:
        mask = mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
    return mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def get_attn_pad_mask(x):
#     batch_size, d_sentence = x.size()
#     pad_attn_mask = (x == 0).unsqueeze(1).repeat(1, d_sentence, 1)
#     return pad_attn_mask
