# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

import json
import logging
import os
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, HfArgumentParser

from lightning_ie.nn import SparseMultiLabelCELossWithLogitsLoss
from lightning_ie.moudle import GlobalPointer
from lightning_ie.util import SequenceDataset, sequence_padding, read_jsonl, write_jsonl

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


def read_schema(file_name):
    schema2id, id2schema = {}, {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            js = json.loads(line)
            schema2id[(js['subject_type'], js['predicate'], js['object_type'])] = i
            id2schema[i] = js
    return schema2id, id2schema


schema2id, id2schema = read_schema('data/CMeIE/53_schemas.jsonl')


def eval_CMeIE(gold_file, pred_file):
    gold = read_jsonl(gold_file)
    pred = read_jsonl(pred_file)

    tp = {i: 0 for i in range(len(schema2id))}
    tot_pred = {i: 0 for i in range(len(schema2id))}
    tot_true = {i: 0 for i in range(len(schema2id))}

    gold_dict = {}
    for example in gold:
        gold_dict[example['text']] = example['spo_list']

        for spo in example['spo_list']:
            tot_true[schema2id[(spo['subject_type'], spo['predicate'], spo['object_type']['@value'])]] += 1

    for example in pred:
        gold_spo_list = gold_dict[example['text']]
        tags = [False] * len(gold_spo_list)
        for spo in example['spo_list']:
            tot_pred[schema2id[(spo['subject_type'], spo['predicate'], spo['object_type']['@value'])]] += 1

            for i, gold_label in enumerate(gold_spo_list):

                if tags[i]:
                    continue

                if (spo['predicate'] == gold_label['predicate']
                        and spo['subject'] == gold_label['subject']
                        and spo['object']['@value'] == gold_label['object']['@value']
                ):
                    tp[schema2id[(spo['subject_type'], spo['predicate'], spo['object_type']['@value'])]] += 1
                    tags[i] = True

    metrics = {}
    for id in id2schema:
        p = tp[id] / max(tot_pred[id], 1e-8)
        r = tp[id] / max(tot_true[id], 1e-8)
        f1 = 2 * p * r / max(p + r, 1e-8)

        metrics[f"{id2schema[id]['subject_type']}_{id2schema[id]['object_type']}_{id2schema[id]['predicate']}_p"] = p
        metrics[f"{id2schema[id]['subject_type']}_{id2schema[id]['object_type']}_{id2schema[id]['predicate']}_r"] = r
        metrics[f"{id2schema[id]['subject_type']}_{id2schema[id]['object_type']}_{id2schema[id]['predicate']}_f1"] = f1

    p = sum([v for k, v in metrics.items() if k.endswith('_p')]) / len(schema2id)
    r = sum([v for k, v in metrics.items() if k.endswith('_r')]) / len(schema2id)
    f1 = 2 * p * r / max(p + r, 1e-8)

    metrics['macro_p'] = p
    metrics['macro_r'] = r
    metrics['macro_f1'] = f1

    p = sum(tp.values()) / max(sum(tot_pred.values()), 1e-8)
    r = sum(tp.values()) / max(sum(tot_true.values()), 1e-8)
    f1 = 2 * p * r / max(p + r, 1e-8)

    metrics['micro_p'] = p
    metrics['micro_r'] = r
    metrics['micro_f1'] = f1

    return metrics


@dataclass()
class CMeIEDataModuleArguments:
    num_workers: int = field(default=0)

    train_batch_size: int = field(default=16)
    valid_batch_size: int = field(default=16)
    test_batch_size: int = field(default=16)


class CMeIEDataModule(pl.LightningDataModule):

    def __init__(self, data, tokenizer, **kwargs):
        super(CMeIEDataModule, self).__init__()
        self.save_hyperparameters(kwargs)

        self.tokenizer = tokenizer
        self.data = data

    def encode_example(self, offset_mapping, example):

        seq_len = len(offset_mapping)
        start_mapping = {j[0]: i for i, j in enumerate(offset_mapping) if j[0] != j[1]}
        end_mapping = {j[1]: i for i, j in enumerate(offset_mapping) if j[0] != j[1]}

        def align(start, end):
            return start_mapping[start], end_mapping[end]

        def search(pattern, sequence):
            n = len(pattern)
            for i in range(len(sequence)):
                if sequence[i:i + n] == pattern:
                    return i, i + n
            return -1

        hrt = set()
        for spo in example['spo_list']:

            head_span = search(spo['subject'], example['text'])
            tail_span = search(spo['object']['@value'], example['text'])

            if head_span == -1 or tail_span == -1:
                continue

            relation = (spo['subject_type'], spo['predicate'], spo['object_type']['@value'])
            p = schema2id[relation]

            try:  # special token
                head_span = align(head_span[0], head_span[1])
                tail_span = align(tail_span[0], tail_span[1])
                hrt.add((head_span, p, tail_span))
            except:
                pass

        mention_ids = [set(), set()]
        head_ids = [set() for _ in range(len(schema2id))]
        tail_ids = [set() for _ in range(len(schema2id))]

        for h, r, t in hrt:
            mention_ids[0].add(h)
            mention_ids[1].add(t)

            head_ids[r].add((h[0], t[0]))
            tail_ids[r].add((h[1], t[1]))

        for label in mention_ids + head_ids + tail_ids:
            if not label:
                label.add((-100, -100))

        mention_ids = sequence_padding([torch.LongTensor(list(_)) for _ in mention_ids], dim=-2)
        head_ids = sequence_padding([torch.LongTensor(list(_)) for _ in head_ids], dim=-2)
        tail_ids = sequence_padding([torch.LongTensor(list(_)) for _ in tail_ids], dim=-2)
        mention_ids = mention_ids[..., 0] * seq_len + mention_ids[..., 1]
        head_ids = head_ids[..., 0] * seq_len + head_ids[..., 1]
        tail_ids = tail_ids[..., 0] * seq_len + tail_ids[..., 1]

        mention_ids[mention_ids < 0] = -100
        head_ids[head_ids < 0] = -100
        tail_ids[tail_ids < 0] = -100

        return mention_ids, head_ids, tail_ids

    def train_collate_fn(self, batch):

        batch_mention_ids = []
        batch_head_ids = []
        batch_tail_ids = []

        inputs = self.tokenizer([_['text'] for _ in batch],
                                padding=True, max_length=512,
                                return_offsets_mapping=True, truncation=True)

        for offset_mapping, example in zip(inputs.offset_mapping, batch):
            mention_ids, head_ids, tail_ids = self.encode_example(offset_mapping, example)
            batch_mention_ids.append(mention_ids)
            batch_head_ids.append(head_ids)
            batch_tail_ids.append(tail_ids)

        inputs = inputs.convert_to_tensors('pt')
        return {
                'inputs': {
                        'input_ids'     : inputs.input_ids,
                        'attention_mask': inputs.attention_mask,
                        'token_type_ids': inputs.token_type_ids,
                },
                'labels': {
                        'mention_ids': sequence_padding(batch_mention_ids),
                        'head_ids'   : sequence_padding(batch_head_ids),
                        'tail_ids'   : sequence_padding(batch_tail_ids),
                }
        }

    def valid_collate_fn(self, batch):
        return [_['text'] for _ in batch]

    def train_dataloader(self):

        return DataLoader(SequenceDataset(self.data['train']),
                          batch_size=self.hparams.train_batch_size,
                          collate_fn=self.train_collate_fn,
                          num_workers=self.hparams.num_workers,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):

        if 'valid' not in self.data:
            return None

        return DataLoader(SequenceDataset(self.data['valid']),
                          batch_size=self.hparams.valid_batch_size,
                          collate_fn=self.valid_collate_fn,
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self):

        if 'test' not in self.data:
            return None

        return DataLoader(SequenceDataset(self.data['test']),
                          batch_size=self.hparams.test_batch_size,
                          collate_fn=self.valid_collate_fn,
                          num_workers=self.hparams.num_workers)


@dataclass()
class ModuleArguments:
    num_labels: int = field(default=len(schema2id))
    learning_rate: float = field(default=5e-5)


class Module(pl.LightningModule):
    def __init__(self, plm_model, tokenizer, **kwargs):
        super(Module, self).__init__()
        self.save_hyperparameters(kwargs)

        self.plm_model = plm_model
        self.tokenizer = tokenizer

        self.hidden_size = self.plm_model.config.hidden_size
        self.mention = GlobalPointer(self.hidden_size, 64, 2, mask_tril=True)
        self.head_idx = GlobalPointer(self.hidden_size, 64, self.hparams.num_labels, use_pe=False)
        self.tail_idx = GlobalPointer(self.hidden_size, 64, self.hparams.num_labels, use_pe=False)

        criterion = SparseMultiLabelCELossWithLogitsLoss()

        def _criterion(input, target):
            B, H, L, L = input.size()
            input = input.reshape(B, H, L * L)
            return criterion(input, target)

        self.criterion = _criterion

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params'      : [p for n, p in param_optimizer if
                                  not any(nd in n for nd in no_decay) and p.requires_grad],
                 'weight_decay': 0.01},
                {'params'      : [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
                 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                      lr=self.hparams.learning_rate)
        return optimizer

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                token_type_ids: torch.LongTensor,
                labels: Dict[str, torch.LongTensor] = None,
                **kwargs,
                ):
        plm_hidden = self.plm_model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    output_hidden_states=True).last_hidden_state

        mention = self.mention(plm_hidden, attention_mask)
        head_idx = self.head_idx(plm_hidden, attention_mask)
        tail_idx = self.tail_idx(plm_hidden, attention_mask)

        output = {'mention_prob': mention,
                  'head_prob'   : head_idx,
                  'tail_prob'   : tail_idx}

        if labels:
            loss_mention = self.criterion(mention, labels['mention_ids'])
            loss_head = self.criterion(head_idx, labels['head_ids'])
            loss_tail = self.criterion(tail_idx, labels['tail_ids'])

            loss = (loss_mention + loss_head + loss_tail) / 3.
            output['loss'] = {'mention_loss': loss_mention,
                              'head_loss'   : loss_head,
                              'tail_loss'   : loss_tail,
                              'total_loss'  : loss}

        return output

    def predict_step(self, batch, batch_idx: int = 0, dataloader_idx: int = 0):

        outputs = self(**batch)

        # mask special tokens [CLS], [SEP]
        batch_mention_prob = outputs['mention_prob']
        batch_head_prob = outputs['head_prob']
        batch_tail_prob = outputs['tail_prob']

        cls_mask = batch.input_ids == self.tokenizer.cls_token_id
        cls_mask = torch.einsum('bm,bn->bmn', cls_mask, cls_mask)
        batch_mention_prob[cls_mask[:, None, :, :].repeat(1, 2, 1, 1)] = -100  # small value
        batch_head_prob[cls_mask[:, None, :, :].repeat(1, self.hparams.num_labels, 1, 1)] = -100
        batch_tail_prob[cls_mask[:, None, :, :].repeat(1, self.hparams.num_labels, 1, 1)] = -100

        sep_mask = batch.input_ids == self.tokenizer.sep_token_id
        sep_mask = torch.einsum('bm,bn->bmn', sep_mask, sep_mask)
        batch_mention_prob[sep_mask[:, None, :, :].repeat(1, 2, 1, 1)] = -100
        batch_head_prob[sep_mask[:, None, :, :].repeat(1, self.hparams.num_labels, 1, 1)] = -100
        batch_tail_prob[sep_mask[:, None, :, :].repeat(1, self.hparams.num_labels, 1, 1)] = -100

        return {'mention_prob': batch_mention_prob,
                'head_prob'   : batch_head_prob,
                'tail_prob'   : batch_tail_prob}

    def validation_step(self, batch, batch_idx: int = 0):

        text = batch
        inputs = self.tokenizer(text, padding=True, truncation=True,
                                max_length=512,
                                return_offsets_mapping=True,
                                return_tensors='pt').to(self.device)
        outputs = self.predict_step(inputs)
        outputs = {k: v.cpu().numpy() for k, v in outputs.items()}
        offset_mapping = inputs.offset_mapping.cpu().numpy()

        examples = []
        for b in range(len(outputs['mention_prob'])):
            example = {'text': text[b], 'spo_list': []}

            heads = set()
            tails = set()
            for _, s, e in zip(*np.where(outputs['mention_prob'][b] > 0)):
                if _ == 0:
                    heads.add((s, e))
                else:
                    tails.add((s, e))

            for mh_s, mh_e in heads:
                for mt_s, mt_e in tails:
                    if (mh_s, mh_e) == (mt_s, mt_e):
                        continue

                    p1 = np.where(outputs['head_prob'][b, :, mh_s, mt_s] > 0)[0]
                    p2 = np.where(outputs['tail_prob'][b, :, mh_e, mt_e] > 0)[0]
                    ps = set(p1) & set(p2)

                    if not ps:
                        continue

                    head_span = int(offset_mapping[b, mh_s, 0]), int(offset_mapping[b, mh_e, 1])
                    tail_span = int(offset_mapping[b, mt_s, 0]), int(offset_mapping[b, mt_e, 1])

                    sbj = text[b][head_span[0]: head_span[1]]
                    obj = text[b][tail_span[0]: tail_span[1]]

                    for p in ps:
                        schema = id2schema[p]
                        example['spo_list'].append({
                                "predicate"   : schema['predicate'],
                                "subject"     : sbj,
                                "subject_type": schema['subject_type'],
                                "object"      : {"@value": obj},
                                "object_type" : {"@value": schema['object_type']}})

            examples.append(example)

        return examples

    def validation_epoch_end(self, outputs):

        # find ModelCheckpoint callback
        dirpath = [_ for _ in self.trainer.callbacks if isinstance(_, pl.callbacks.ModelCheckpoint)][0].dirpath
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        outputs = sum(outputs, start=[])
        global_step = self.trainer.global_step
        current_epoch = self.trainer.current_epoch

        log_file_path = os.path.join(dirpath, f'log.txt')
        json_file_path = os.path.join(dirpath, f'{current_epoch:02d}-{global_step:06d}.jsonl')

        write_jsonl(json_file_path, outputs)
        metrics = eval_CMeIE('data/CMeIE/CMeIE_dev.jsonl', json_file_path)

        logger.info(pformat(metrics))
        self.log_dict(metrics)

        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(
                    f"{current_epoch:02d}-{global_step:06d}"
                    f":{metrics}\n")

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):

        # find ModelCheckpoint callback
        dirpath = [_ for _ in self.trainer.callbacks if isinstance(_, pl.callbacks.ModelCheckpoint)][0].dirpath
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        outputs = sum(outputs, start=[])
        json_file_path = os.path.join(dirpath, f'test.jsonl')

        write_jsonl(json_file_path, outputs)

    def training_step(self, batch, batch_idx):
        loss = self(**batch['inputs'], labels=batch['labels'])['loss']
        self.log_dict(loss, prog_bar=True)
        return loss['total_loss']


def main(module_args, data_module_args, trainer_args):
    # init model
    config = AutoConfig.from_pretrained('Langboat/mengzi-bert-L6-H768')
    plm_model = AutoModel.from_pretrained('Langboat/mengzi-bert-L6-H768', config=config)
    tokenizer = AutoTokenizer.from_pretrained('Langboat/mengzi-bert-L6-H768', config=config)

    # load data
    train_data = read_jsonl('data/CMeIE/CMeIE_train.jsonl')
    valid_data = read_jsonl('data/CMeIE/CMeIE_dev.jsonl')
    test_data = read_jsonl('data/CMeIE/CMeIE_test.jsonl')

    # init module
    module = Module(plm_model, tokenizer, **vars(module_args))
    dataset = CMeIEDataModule({'train': train_data, 'valid': valid_data, 'test': test_data},
                              tokenizer, **vars(data_module_args))
    callback = ModelCheckpoint(monitor='micro_f1',
                               filename='{epoch:02d}-{step:06d}-{micro_f1:.4f}',
                               save_top_k=1,
                               mode='max')
    trainer = Trainer.from_argparse_args(trainer_args, callbacks=[callback])

    trainer.fit(module, dataset)
    module = module.load_from_checkpoint(callback.best_model_path,
                                         plm_model=plm_model,
                                         tokenizer=tokenizer,
                                         **vars(module_args))
    trainer.validate(module, dataset)
    trainer.test(module, dataset)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = HfArgumentParser([ModuleArguments, CMeIEDataModuleArguments])
    parser = Trainer.add_argparse_args(parser)

    module_args, data_module_args, trainer_args = parser.parse_args_into_dataclasses()

    seed_everything(42, workers=True)
    main(module_args, data_module_args, trainer_args)
