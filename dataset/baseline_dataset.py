"""BART baseline dataset that has no drafts, for seq2seq and kpseq2seq"""
import json
import torch
from tqdm import tqdm
from .base_dataset import BaseDataset
from nltk.tree import Tree
import utils

from torchtext.vocab import Vocab
from collections import Counter
fully_label_vocab = Vocab(  # from TRAINING (TRAIN+DEV)
      Counter(
          [
              # N-S
              "nucleus-satellite:Elaboration",
              "nucleus-satellite:Attribution",
              "nucleus-satellite:Explanation",
              "nucleus-satellite:Enablement",
              "nucleus-satellite:Background",
              "nucleus-satellite:Evaluation",
              "nucleus-satellite:Cause",
              "nucleus-satellite:Contrast",
              "nucleus-satellite:Condition",
              "nucleus-satellite:Comparison",
              "nucleus-satellite:Manner-Means",
              "nucleus-satellite:Summary",
              "nucleus-satellite:Temporal",
              "nucleus-satellite:Topic-Comment",
              "nucleus-satellite:Topic-Change",
              # S-N
              "satellite-nucleus:Attribution",
              "satellite-nucleus:Contrast",
              "satellite-nucleus:Background",
              "satellite-nucleus:Condition",
              "satellite-nucleus:Cause",
              "satellite-nucleus:Evaluation",
              "satellite-nucleus:Temporal",
              "satellite-nucleus:Explanation",
              "satellite-nucleus:Enablement",
              "satellite-nucleus:Comparison",
              "satellite-nucleus:Elaboration",
              "satellite-nucleus:Manner-Means",
              "satellite-nucleus:Summary",
              "satellite-nucleus:Topic-Comment",
              # N-N
              "nucleus-nucleus:Joint",
              "nucleus-nucleus:Same-unit",
              "nucleus-nucleus:Contrast",
              "nucleus-nucleus:Temporal",
              "nucleus-nucleus:Topic-Change",
              "nucleus-nucleus:Textual-organization",
              "nucleus-nucleus:Comparison",
              "nucleus-nucleus:Topic-Comment",
              "nucleus-nucleus:Cause",
              "nucleus-nucleus:Condition",
              "nucleus-nucleus:Explanation",
              "nucleus-nucleus:Evaluation",
          ]
      ),
      specials=["<pad>"],
  )

class BaselineDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path = f'data_rst/rst_{self.set_type}.jsonl'
        #path = f'rst_mini.jsonl'
        #path = f'rst_test.jsonl'
        #path = f'test_100.jsonl'
        #path = f'test_10.jsonl'
        self.load_raw_dataset(path=path)


    def load_raw_dataset(self, path):
        """Load raw data for baseline models."""

        print(f'loading {path}')
        for ln in tqdm(open(path)):
            cur_obj = json.loads(ln)
            tgt_tokens = cur_obj['tgt_tokens']
            if len(tgt_tokens) > 500:
                continue
            cur_tgt_ids = cur_obj['tgt_tokens'][2:-1]
            cur_tgt_ids = cur_tgt_ids[:self.max_tgt_len]
            
            edus = []
            _edu = []
            
            for i, tok in enumerate(cur_tgt_ids):
                if tok == 0:
                    edus.append(_edu)
                    _edu = []
                else:
                    _edu.append(tok)
            edus.append(_edu)

            
            _discourse_label = cur_obj['label']
            discourse_label = [self.mask_idx]*len(edus[0])
            for i,edu in enumerate(edus[1:]):
                _label = _discourse_label[i]
                if _label == None:
                    label = self.mask_idx
                else:
                    label = fully_label_vocab[_label]
                discourse_label.extend([label]*len(edu))
            if any([label == None for label in discourse_label]):
                continue
            discourse_label = discourse_label + [label]
            self.discourse_labels.append(discourse_label)
 
            cur_src = cur_obj['kp_set_str']
            cur_src_ids = self.tokenizer.encode(
                ' '+cur_src,
                max_length=self.max_kp_len,
                truncation=True,
                add_special_tokens=False,
            )
            cur_src_ids = [self.bos_idx] + cur_src_ids + [self.eos_idx]
            self.source.append(cur_src_ids)


            
            tgt = []
            for edu in edus:
                for token in edu:
                    tgt.append(token)
            tgt = [self.bos_idx] + tgt + [self.eos_idx]
            self.target.append(tgt)
            
            disc_len = [len(edu) for edu in edus]
            self.discourse_lengths.append(disc_len)
            pos = []
            for i, edu in enumerate(edus):
                pos.extend([i]*len(edu))
            pos = pos + [pos[-1]]
            self.discourse_positions.append(pos)
            
            

            self.ID.append(cur_obj['id'])
            
            kp_plan_str = cur_obj['kp_plan_str']
            kp_plan_list = kp_plan_str.split(' <s> ')
            bag_of_words = []
            for kp_plan in kp_plan_list:
                if kp_plan == '':
                    bag_of_words.append([])
                else:
                    kp_plan = self.tokenizer.encode(
                        ' '+kp_plan,
                        max_length=self.max_kp_len,
                        truncation=True,
                        add_special_tokens=False,
                    )
                    kp_plan = [[kp] for kp in kp_plan]
                    bag_of_words.append(kp_plan)
            self.bow.append(bag_of_words)
                                                    
            


    def __getitem__(self, index):
        cur_id = self.ID[index]
        cur_src_ids = self.source[index]
        input_ids = torch.LongTensor(cur_src_ids)
        cur_tgt_ids = self.target[index]
        tgt_ids = torch.LongTensor(cur_tgt_ids)
        discourse_pos = self.discourse_positions[index]
        discourse_pos = torch.LongTensor(discourse_pos)
        discourse_label = self.discourse_labels[index]
        discourse_label = torch.LongTensor(discourse_label)
        discourse_length = self.discourse_lengths[index]
        bow = self.bow[index]

        ret_obj = dict(
            id=cur_id,
            input_ids=input_ids,
            tgt_ids=tgt_ids,
            discourse_pos=discourse_pos,
            discourse_label=discourse_label,
            discourse_length=discourse_length,
            bow=bow,
        )
        
        return ret_obj

