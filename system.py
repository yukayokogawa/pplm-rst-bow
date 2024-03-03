import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from argparse import Namespace
import math
from collections import OrderedDict
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss

from transformers import (
    AdamW,
    BartTokenizer,
    get_linear_schedule_with_warmup,
)
#from torch_scatter import scatter_mean

from dataset import BaselineDataset
from models import Discriminator

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


class DiscourseClassifier(pl.LightningModule):

    def __init__(self, hparams, is_inference=False):
        """
        Args:
            hparams (argparse.Namespace): hyper-parameters, domain, and setup information
            is_inference (bool): True for decoding, False for train/valid
        """
        super().__init__()
        self.is_inference = is_inference
        #print(hparams)
        
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        
        self.save_hyperparameters(hparams)
        
        self.relation_num = len(fully_label_vocab)

        self.discriminator = Discriminator(self.relation_num)

        

        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        
        
        self.dataset_cls = BaselineDataset

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.discriminator
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer

        return [optimizer]


    def optimizer_step(self, epoch, batch_idx, optimizer,
                       opt_idx, lambda_closure, using_native_amp
                       ):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()


    def forward(self, input_ids, discourse_pos=None):
        return self.discriminator(
            input_ids,
            discourse_pos=discourse_pos,
        )

    def train_dataloader(self):
        train_set = self.dataset_cls(
            args=self.hparams,
            set_type=self.hparams.train_set,
            tokenizer=self.tokenizer,
            is_inference=False
        )

        dataloader = DataLoader(train_set, batch_size=self.hparams.train_batch_size,
                                collate_fn=train_set.collater)

        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpus)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = self.dataset_cls(
            args=self.hparams,
            set_type=self.hparams.valid_set,
            tokenizer=self.tokenizer,
            is_inference=False
        )
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size,
                          collate_fn=val_dataset.collater)

    def test_dataloader(self, chunk_id='all', saved_ids=[]):
        test_dataset = self.dataset_cls(
            args=self.hparams,
            set_type=self.hparams.test_set,
            tokenizer=self.tokenizer,
            is_inference=True,
        )

        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size,
                          collate_fn=test_dataset.collater)

    def _step(self, batch):
        rel_logits = self(**batch['net_input'])
        #print('logits : ', rel_logits.shape)
        target = batch['discourse_labels']
        #print(target)
        #print('target : ',target.shape)
        target = target.view(-1)
        rel_loss_fct = CrossEntropyLoss(ignore_index=-100)
        rel_loss = rel_loss_fct(rel_logits.view(-1, self.relation_num), target)
        
        predictions = torch.argmax(rel_logits, dim=-1)
        preds_ = predictions.view(-1)
        labels_ = batch['discourse_labels'].view(-1)
        labels = labels_[labels_.ne(-100)]
        preds = preds_[labels_.ne(-100)]
        #print(preds)

        corr = preds.eq(labels)
        accuracy = sum(corr).item() / len(corr)
        
        # loss = outputs[0]
        loss = rel_loss
        ppl = math.exp(loss)
        ppl = torch.Tensor([ppl]).to(loss.device)
        accuracy = torch.Tensor([accuracy]).to(loss.device)

        return loss, ppl, accuracy


    def training_step(self, batch, batch_idx):
        loss, ppl, acc = self._step(batch)
        cur_lr = self.lr_scheduler.get_last_lr()[0]

        if self.hparams.n_gpus > 1:
            loss = loss.unsqueeze(0)
            cur_lr = torch.Tensor([cur_lr]).to(loss.device)

        result = pl.TrainResult(minimize=loss)
        result.log("lr", torch.Tensor([cur_lr]), on_step=True, on_epoch=True)
        result.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        result.log("train_ppl", ppl, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        result.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return result
        

    def test_step(self, batch, batch_idx):
        loss, ppl, acc = self._step(batch)
        output = OrderedDict({
            'test_loss': loss,
            'test_ppl': ppl,
            'test_acc': acc,
        })
        return output
    

    def validation_step(self, batch, batch_idx):
        loss, ppl, acc = self._step(batch)
        output = OrderedDict({
            'val_loss': loss,
            'val_ppl': ppl,
            'val_acc': acc,
        })
        return output


    def validation_epoch_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0
        val_ppl_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']
            val_ppl_mean += output['val_ppl']
        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        val_ppl_mean /= len(outputs)

        result = pl.EvalResult(checkpoint_on=val_loss_mean)
        result.log("val_loss", val_loss_mean, prog_bar=True, logger=True)
        result.log("val_acc", val_acc_mean, prog_bar=True, logger=True)
        result.log("val_ppl", val_ppl_mean, prog_bar=True, logger=True)
        return result
    
    def test_epoch_end(self, outputs):
        test_loss_mean = 0
        test_acc_mean = 0
        test_ppl_mean = 0
        for output in outputs:
            test_loss_mean += output['test_loss']
            test_acc_mean += output['test_acc']
            test_ppl_mean += output['test_ppl']
        test_loss_mean /= len(outputs)
        test_acc_mean /= len(outputs)
        test_ppl_mean /= len(outputs)

        result = pl.EvalResult()
        result.log("test_loss", test_loss_mean, prog_bar=True, logger=True)
        result.log("test_acc", test_acc_mean, prog_bar=True, logger=True)
        result.log("test_ppl", test_ppl_mean, prog_bar=True, logger=True)
        return result
        
