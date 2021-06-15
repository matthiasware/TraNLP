import numpy as np
import torch
from torch.utils.data import Dataset


def get_specialized_vocabulary(vocabulary):
    spec_tok_dict = {'[PAD]': 0, '[MSK]': 1, '[CLS]': 2}
    tok_dict = {}

    # word dict
    for tok, idx in spec_tok_dict.items():
        tok_dict[tok] = idx
    for idx, tok in enumerate(vocabulary):
        tok_dict[tok] = idx + len(spec_tok_dict)

    return tok_dict


class GrammarDataset(Dataset):
    def __init__(self, samples, tok_dict, d_sentence,
                 pred_min=1, pred_max=1, pred_freq=0.15):
        self.samples = samples
        self.tok_dict = tok_dict
        self.pred_min = pred_min
        self.pred_max = pred_max
        self.pred_freq = pred_freq
        self.d_sentence = d_sentence
        self.idx_dict = {v: k for k, v in tok_dict.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sentence = self.samples[idx]

        # sentence to idx vectors
        # -) tokenize
        tokens = list(sentence)

        # -) replace with vocabulary idcs
        tok_list = [self.tok_dict[tok] for tok in tokens]
        tok_list = np.array(tok_list)

        # -) calculate the number of predctions
        n_preds = int(round(len(tok_list) * self.pred_freq))
        n_preds = min(max(self.pred_min, n_preds), self.pred_max)

        # -) create MASKS
        mask_idcs = np.random.choice(
            len(tok_list), size=n_preds, replace=False)
        mask_toks = tok_list[mask_idcs]
        tok_list[mask_idcs] = self.tok_dict["[MSK]"]
        # add 1 to mask idxs since we added one position in the front
        mask_idcs += 1
        np.pad(mask_toks, (0, self.pred_max), mode="constant")

        # -) PAD
        n_pad = self.d_sentence - len(tok_list)
        tok_list = np.pad(tok_list, (1, n_pad - 1), mode='constant')

        # ADD CLS Token to start
        tok_list[0] = self.tok_dict['[CLS]']

        # torchify
        tok_list = torch.LongTensor(tok_list)
        mask_idcs = torch.LongTensor(mask_idcs)
        mask_toks = torch.LongTensor(mask_toks)

        return tok_list, mask_idcs, mask_toks
