import torch
import torch.nn.functional as F
from .strategy_utils import (
    top_k_top_p_filtering,
    make_past_pplm,
    make_past_kv_bart,
)

from .pplm import perturb_past
import utils

from decoding_strategy import BaseDecoding

import numpy as np

BIG_CONST = 1e10
SMALL_CONST = 1e-15

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}


def build_bows_one_hot_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        #print(single_bow)
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors

def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


class PPLMDecoding(BaseDecoding):

    def generate(self, model, classifier, proj, batch):
        
        

        net_input = utils.move_to_cuda(batch['net_input'])
        encoder_input_ids = net_input['input_ids']
        encoder_attn_mask = net_input['attention_mask']
        batch_size = encoder_input_ids.shape[0]

        encoder = model.get_encoder()
        encoder_outputs = encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attn_mask,
        )
        

        # create empty decoder_input_ids
        input_ids = torch.full(
            (batch_size, 1),
            self.decoder_bos_idx,
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        cur_len = 1
        
        probs = [[] for _ in range(batch_size)]

        unfinished_sents = input_ids.new(batch_size).fill_(1)
        
        disc_lens = batch['discourse_lengths']
        disc_lens = disc_lens[0]
        #print('disc_lens', disc_lens)
        edu_len_sum = 0
        edu_end_pos = []
        for length in disc_lens:
            edu_len_sum = edu_len_sum + length
            edu_end_pos.append(edu_len_sum)
        #print('edu_end_pos', edu_end_pos)
        max_len = sum(disc_lens)+1
        #print('max_len', max_len)
        bows = batch['bow']
        bows = bows[0]
        bag_of_words = []
        
        for i,bow_edu in enumerate(bows):
            #print('i',i)
            #print('bow_edu', bow_edu)
            if bow_edu == []:
                one_hot_bows_vectors = []
            else:
                one_hot_bows_vectors = build_bows_one_hot_vectors([bow_edu], self.tokenizer)
            bag_of_words.append(one_hot_bows_vectors)
        discourse_labels = batch['discourse_labels']
        discourse_labels = discourse_labels[0]
        #print('discourse_labels',discourse_labels)
        discourse_pos = batch['discourse_pos']
        discourse_pos = discourse_pos[0]
        #print('discourse_pos', discourse_pos)
        past = None 
        past_kv = None
        
        grad_norms = None
        last = input_ids[:,-1:]
        
        max_tgt_len = 200
        stepsize = self.stepsize
        gm_scale = self.gm_scale
        kl_scale = self.kl_scale
        

        
        previous_accumulated_hidden = None
        while cur_len < max_tgt_len:
            perturb = False
            try:
                cur_edu_num = discourse_pos[cur_len-1]
            except:
                cur_edu_num = -1
            if cur_edu_num == 63:
                cur_edu_num = -1
            
            #print('cur_edu_num',cur_edu_num)
            if cur_edu_num >= 1:
                previous_edu_len = disc_lens[cur_edu_num-1]
            else:
                previous_edu_len = 0
            if cur_edu_num >= 1:
                previous_edu_end_pos = edu_end_pos[cur_edu_num-1]
            else:
                previous_edu_end_pos = 0
            if cur_edu_num >= 0:
                cur_edu_len = cur_len - 1 - previous_edu_end_pos
                cur_label = discourse_labels[cur_len-1]
                cur_bow = bag_of_words[cur_edu_num]
                """
                if cur_bow != []:
                    print(torch.nonzero(cur_bow[0]))
                """
                
            """
            if past is None and past_kv is None:
                outputs = model(
                    decoder_input_ids=input_ids,
                    encoder_outputs=encoder_outputs,
                    attention_mask=encoder_attn_mask,
                )
                past_kv = outputs['past_key_values']
 
                past = make_past_pplm(past_kv)
            """
            """
            inputs = model.prepare_inputs_for_generation(
                input_ids,
                past=past_kv,
                attention_mask=encoder_attn_mask,
                encoder_outputs=encoder_outputs,
            )
            """
            
            """
            unpert_outputs = model(
                decoder_input_ids=last,
                past_key_values=past_kv,
                encoder_outputs=encoder_outputs,
                #attention_mask=encoder_attn_mask,
            )
            """
            
            unpert_outputs = model(
                decoder_input_ids=input_ids,
                encoder_outputs=encoder_outputs,
                #attention_mask=encoder_attn_mask,
            )
            
            #unpert_outputs = model(**inputs)
            unpert_logits = unpert_outputs['logits']
            unpert_past_kv = unpert_outputs['past_key_values']
            unpert_past = make_past_pplm(unpert_past_kv)
            unpert_all_hidden = unpert_outputs['decoder_hidden_states']
            unpert_last_hidden = unpert_all_hidden[-1]
            #unpert_token = torch.argmax(unpert_logits[:,-1], dim=-1)
            #print('unpert : ',unpert_token)
            
            if cur_edu_num >= 1:

                previous_edu_start = edu_end_pos[cur_edu_num-2]+1 if cur_edu_num >= 2 else 0
                previous_edu_end = edu_end_pos[cur_edu_num-1]
                previous_accumulated_hidden = pert_last_hidden[:,previous_edu_start:previous_edu_end+1]
                previous_accumulated_hidden = torch.sum(previous_accumulated_hidden, dim=1)
                accumulated_hidden = unpert_last_hidden[:, previous_edu_end+1:-1, :]
                accumulated_hidden = torch.sum(accumulated_hidden, dim=1)
            else:
                accumulated_hidden = unpert_last_hidden[:, :-1, :]
                accumulated_hidden = torch.sum(accumulated_hidden, dim=1)
            
            
            if (past is not None) and (not self.no_rst) and (not self.no_bow) and cur_edu_num >= 0 and (cur_label not in [self.mask_idx, 0]) and cur_bow != []:
                pert_past, pert_past_kv,  _, grad_norms, _ = perturb_past(
                    past,
                    model,
                    last,
                    past_kv=past_kv,
                    unpert_past=unpert_past,
                    unpert_past_kv=unpert_past_kv,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    previous_accumulated_hidden=previous_accumulated_hidden,
                    previous_edu_len=previous_edu_len,
                    cur_edu_len=cur_edu_len,
                    encoder_outputs=encoder_outputs,
                    encoder_attn_mask=encoder_attn_mask,
                    grad_norms=grad_norms,
                    stepsize=stepsize,
                    one_hot_bows_vectors=cur_bow,
                    classifier=classifier,
                    class_label=cur_label,
                    loss_type=3,
                    num_iterations=3,
                    #num_iterations=30,
                    #horizon_length=1,
                    horizon_length=0,
                    window_length=cur_edu_len,
                    decay=False,
                    #gamma=1.5,
                    gamma=1.5,
                    kl_scale=kl_scale,
                    device='cuda',
                    verbosity_level=1,
                    proj=proj,
                )
                
            elif (past is not None) and (not self.no_rst) and cur_edu_num >= 0 and cur_label not in [self.mask_idx, 0]:
                pert_past, pert_past_kv,  _, grad_norms, _ = perturb_past(
                    past,
                    model,
                    last,
                    past_kv=past_kv,
                    unpert_past=unpert_past,
                    unpert_past_kv=unpert_past_kv,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    previous_accumulated_hidden=previous_accumulated_hidden,
                    previous_edu_len=previous_edu_len,
                    cur_edu_len=cur_edu_len,
                    encoder_outputs=encoder_outputs,
                    encoder_attn_mask=encoder_attn_mask,
                    grad_norms=grad_norms,
                    stepsize=stepsize,
                    one_hot_bows_vectors=None,
                    classifier=classifier,
                    class_label=cur_label,
                    loss_type=2,
                    num_iterations=3,
                    #num_iterations=30,
                    #horizon_length=1,
                    horizon_length=0,
                    window_length=cur_edu_len,
                    decay=False,
                    #gamma=1.5,
                    gamma=1.0,
                    kl_scale=kl_scale,
                    device='cuda',
                    verbosity_level=1,
                    proj=proj,
                )
                
            elif (past is not None) and (not self.no_bow) and cur_edu_num >= 0 and cur_bow != []:
                pert_past, pert_past_kv,  _, grad_norms, _ = perturb_past(
                    past,
                    model,
                    last,
                    past_kv=past_kv,
                    unpert_past=unpert_past,
                    unpert_past_kv=unpert_past_kv,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    previous_accumulated_hidden=previous_accumulated_hidden,
                    previous_edu_len=previous_edu_len,
                    cur_edu_len=cur_edu_len,
                    encoder_outputs=encoder_outputs,
                    encoder_attn_mask=encoder_attn_mask,
                    grad_norms=grad_norms,
                    stepsize=stepsize,
                    #stepsize=0.1,
                    one_hot_bows_vectors=cur_bow,
                    classifier=None,
                    class_label=None,
                    loss_type=1,
                    num_iterations=3,
                    #horizon_length=1,
                    horizon_length=0,
                    window_length=cur_edu_len,
                    decay=False,
                    gamma=1.5,
                    kl_scale=kl_scale,
                    device='cuda',
                    verbosity_level=1,
                    proj=proj,
                )
                
            else:
                #print('not perturb')
                pert_past = past
                
                pert_past_kv = past_kv
                grad_norms = grad_norms
            
            inputs = model.prepare_inputs_for_generation(
                input_ids,
                past=pert_past_kv,
                attention_mask=encoder_attn_mask,
                encoder_outputs=encoder_outputs,
            )
            """
            pert_outputs = model(
                decoder_input_ids=last,
                past_key_values=pert_past_kv,
                encoder_outputs=encoder_outputs,
                #attention_mask=encoder_attn_mask,
            )
            """
            pert_outputs = model(**inputs)
            pert_logits = pert_outputs['logits']
            #print(pert_logits)
            
            
            past_kv = pert_outputs['past_key_values']
            #past_kv = unpert_past_kv
            #past = unpert_past
            
            past = make_past_pplm(past_kv)
            pert_all_hidden = pert_outputs['decoder_hidden_states']
            pert_last_hidden = pert_all_hidden[-1]
            #print('pert_all_hidden',pert_all_hidden[0])
            if gm_scale > 0:
                pert_logits = pert_logits[:, -1, :] / self.temperature  #+ SMALL_CONST
                #print(pert_logits)
                #pert_logits[:,cur_len-1 <= max_len-1, self.eos_idx] = -10000.
                unpert_logits = unpert_logits[:, -1, :] / self.temperature #+ SMALL_CONST
                #unpert_logits[:,cur_len-1 <= max_len-1, self.eos_idx] = -10000.

                pert_probs = F.softmax(pert_logits, dim=-1)
                unpert_probs = F.softmax(unpert_logits, dim=-1)
                #print('pert_probs',pert_probs)
                #print('unpert_probs',unpert_probs)
                if cur_edu_num >= 0:
                    next_token_probs = ((pert_probs ** gm_scale) * (
                            unpert_probs ** (1 - gm_scale)))   #+ SMALL_CONST
                else:
                    next_token_probs = pert_probs
                next_token_probs = top_k_top_p_filtering(next_token_probs, top_k=self.topk, top_p=self.topp, probs=True)

                # rescale
                """
                if torch.sum(pert_probs) <= 1:
                    print('!')
                    next_token_probs = next_token_probs / torch.sum(next_token_probs)
                """
                #print(pert_probs)
                next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
            else:
                
                #print('!')
                if self.do_sampling:
                    next_token_logits = pert_logits[:,-1]
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if self.temperature != 1.0:
                        next_token_logits = next_token_logits / self.temperature
                    # Top-p/top-k filtering
                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=self.topk, top_p=self.topp)
                    # Sample
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.argmax(next_token_logits, dim=-1)
            

            #chosen_token_probs = next_token_probs.gather(1, next_token.view(-1, 1))
            chosen_token_probs = next_token_probs.gather(1, next_token.view(-1, 1))

            #print(chosen_token_probs)
            for b in range(batch_size):
                probs[b].append(chosen_token_probs[b,0].item())

            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (self.pad_idx) * (1 - unfinished_sents)

            if not self.quiet:
                output_str = ''
                for b in range(batch_size):
                    w = self.tokenizer.convert_ids_to_tokens([tokens_to_add[b]])[0]
                    p = probs[b][-1]
                    output_str += '{:>12}({:.2f})|'.format(w, 100 * p)
                if cur_len == 1:
                    print('=' * 50)
                print('step={:<3d}|{}'.format(cur_len, output_str))

            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            eos_in_sents = tokens_to_add == self.eos_idx
            unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break
            cur_len = cur_len + 1
            last = input_ids[:, -1:]

        return input_ids, probs


