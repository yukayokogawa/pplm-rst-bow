from torch.utils.data import Dataset
import utils
import numpy as np
import torch

# (op, kp, tgt)
DOMAIN_TO_MAX_SIZES = {
    'arggen': [33, 100, 140],
    'opinion': [13, 150, 243],
    'news': [15, 250, 335],
    'rst' : [200,100,500],
}

class BaseDataset(Dataset):

    def __init__(self, args, set_type, tokenizer, is_inference):
        super().__init__()

        self.domain = args.domain
        self.setup = args.setup
        self.tokenizer = tokenizer
        self.set_type = set_type
        self.is_inference = is_inference

        self.max_prompt_len = args.max_prompt_len if args.max_prompt_len is not None else DOMAIN_TO_MAX_SIZES[self.domain][0]
        self.max_kp_len = args.max_kp_len if args.max_kp_len is not None else DOMAIN_TO_MAX_SIZES[self.domain][1]
        self.max_tgt_len = args.max_tgt_len if args.max_tgt_len is not None else DOMAIN_TO_MAX_SIZES[self.domain][2]


        self.max_disc_len = 64

        self.sep_tok = '<s>'
        self.sep_idx = 0

        self.bok_tok = '<s>'
        self.bok_idx = 0

        self.bos_tok = '<s>'
        self.bos_idx = 0

        self.pad_tok = '<pad>'
        self.pad_idx = 1

        self.mask_tok = '<mask>'
        self.mask_idx = 50264

        self.eos_tok = '</s>'
        self.eos_idx = 2

        self.ID = []
        self.source = []
        self.target = []
        self.discourse_lengths = []
        self.discourse_positions = []
        self.discourse_labels = []
        
        self.bow = []
        
        self.random = np.random.RandomState(42)

    def __len__(self):
        return len(self.ID)

    def load_raw_dataset(self, path):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def collater(self, samples):
        def merge(key, is_list=False, pad_idx=1):
            if is_list:
                res = []
                for i in range(len(samples[0][key])):
                    res.append(utils.collate_tokens(
                        [s[key][i] for s in samples], pad_idx=pad_idx,
                    ))
                return res
            else:
                return utils.collate_tokens(
                    [s[key] for s in samples], pad_idx=pad_idx,
                )
            
        input_ids = merge('input_ids')
        attention_mask = input_ids.ne(1).long()
        
        net_input = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        y = merge('tgt_ids')
        dec_in = y[:, :-1].contiguous()
        dec_in[dec_in == self.eos_idx] = self.pad_idx
        lm_labels = y[:, 1:].clone()
        lm_labels[lm_labels == self.pad_idx] = -100
        net_input['decoder_input_ids'] = dec_in
   
        
        discourse_pos = merge('discourse_pos', pad_idx=(self.max_disc_len-1))
        if len(samples) == 1:
            pad_tensor = torch.LongTensor([self.pad_idx]).unsqueeze(0)
            dec_in = torch.cat((dec_in, pad_tensor), dim=1)
            pos_pad_tensor = torch.LongTensor([self.max_disc_len-1]).unsqueeze(0)
            discourse_pos = torch.cat((discourse_pos, pos_pad_tensor), dim=1)
        """
        _discourse_labels = [s['discourse_label'] for s in samples]
        discourse_labels = []
        for label in _discourse_labels:
            new_label = label + [-100]*(self.max_disc_len-1-len(label))
            discourse_labels.append(new_label)
        discourse_labels = torch.LongTensor(discourse_labels)
        """
        discourse_labels = merge('discourse_label', pad_idx=0)
        ret_obj = dict(
            id=[s['id'] for s in samples],
            lm_labels=lm_labels,
            discourse_labels=discourse_labels,
            discourse_lengths=[s['discourse_length'] for s in samples],
            bow=[s['bow'] for s in samples],
            discourse_pos=discourse_pos,
        )
        
        
        ret_obj['net_input'] = net_input

        return ret_obj
