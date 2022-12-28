# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

import logging
import os
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, HfArgumentParser

from lightning_ie.util import SequenceDataset, sequence_padding, read_json, write_json, decode_bio_seq

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


def read_schema():
    schema2id = {
            'O'    : 0,
            'B-dis': 1, 'I-dis': 2,
            'B-sym': 3, 'I-sym': 4,
            'B-pro': 5, 'I-pro': 6,
            'B-equ': 7, 'I-equ': 8,
            'B-dru': 9, 'I-dru': 10,
            'B-ite': 11, 'I-ite': 12,
            'B-bod': 13, 'I-bod': 14,
            'B-dep': 15, 'I-dep': 16,
            'B-mic': 17, 'I-mic': 18,
    }
    id2schema = {k: v for v, k in schema2id.items()}

    return schema2id, id2schema


schema2id, id2schema = read_schema()


def eval_CMeEE(gold_file, pred_file):
    gold = read_json(gold_file)
    pred = read_json(pred_file)

    tp = {i: 0 for i in range(len(schema2id) // 2)}
    tot_pred = {i: 0 for i in range(len(schema2id) // 2)}
    tot_true = {i: 0 for i in range(len(schema2id) // 2)}

    gold_dict = {}
    for example in gold:
        gold_dict[example['text']] = example['entities']

        for entity in example['entities']:
            tot_true[schema2id['B-' + entity['type']] // 2] += 1

    for example in pred:
        gold_entities = gold_dict[example['text']]
        tags = [False] * len(gold_entities)
        for entity in example['entities']:
            tot_pred[schema2id['B-' + entity['type']] // 2] += 1

            for i, gold_label in enumerate(gold_entities):

                if tags[i]:
                    continue

                if (entity['type'] == gold_label['type']
                        and entity['start_idx'] == gold_label['start_idx']
                        and entity['end_idx'] == gold_label['end_idx']
                ):
                    tp[schema2id['B-' + entity['type']] // 2] += 1
                    tags[i] = True

    metrics = {}
    for id in tp:
        p = tp[id] / max(tot_pred[id], 1e-8)
        r = tp[id] / max(tot_true[id], 1e-8)
        f1 = 2 * p * r / max(p + r, 1e-8)

        t = id2schema[id * 2 + 1].split('-')[1]
        metrics[f"{t}_p"] = p
        metrics[f"{t}_r"] = r
        metrics[f"{t}_f1"] = f1

    p = sum([v for k, v in metrics.items() if k.endswith('_p')]) / (len(schema2id) // 2)
    r = sum([v for k, v in metrics.items() if k.endswith('_r')]) / (len(schema2id) // 2)
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
class CMeEEDataModuleArguments:
    num_workers: int = field(default=0)

    train_batch_size: int = field(default=16)
    valid_batch_size: int = field(default=16)
    test_batch_size: int = field(default=16)


class CMeEEDataModule(pl.LightningDataModule):

    def __init__(self, data, tokenizer, **kwargs):
        super(CMeEEDataModule, self).__init__()
        self.save_hyperparameters(kwargs)

        self.tokenizer = tokenizer
        self.data = data

    def encode_example(self, offset_mapping, example):

        seq_len = len(offset_mapping)
        start_mapping = {j[0]: i for i, j in enumerate(offset_mapping) if j[0] != j[1]}
        end_mapping = {j[1]: i for i, j in enumerate(offset_mapping) if j[0] != j[1]}

        mask_len = 0
        for s, e in offset_mapping[::-1]:
            if s != e:
                break
            mask_len += 1

        def align(start, end):
            return start_mapping[start], end_mapping[end]

        bio_seq = [-100] + [schema2id['O']] * (seq_len - mask_len - 1) + [-100] * mask_len
        for entity in example['entities']:

            head_span = entity['start_idx']
            tail_span = entity['end_idx'] + 1
            entity_type = entity['type']

            try:  # special token
                head_span, tail_span = align(head_span, tail_span)
            except:
                continue

            bio_seq[head_span] = schema2id[f"B-{entity_type}"]
            for i in range(head_span + 1, tail_span + 1):
                bio_seq[i] = schema2id[f"I-{entity_type}"]

        return torch.LongTensor(bio_seq)

    def train_collate_fn(self, batch):

        batch_bio_seq = []

        inputs = self.tokenizer([_['text'] for _ in batch],
                                padding=True, max_length=512,
                                return_offsets_mapping=True, truncation=True)

        for offset_mapping, example in zip(inputs.offset_mapping, batch):
            bio_seq = self.encode_example(offset_mapping, example)
            batch_bio_seq.append(bio_seq)

        inputs = inputs.convert_to_tensors('pt')
        return {
                'inputs': {
                        'input_ids'     : inputs.input_ids,
                        'attention_mask': inputs.attention_mask,
                        'token_type_ids': inputs.token_type_ids,
                },
                'labels': {
                        'bio_seq': sequence_padding(batch_bio_seq),
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
    learning_rate: float = field(default=2e-5)


class Module(pl.LightningModule):
    def __init__(self, plm_model, tokenizer, **kwargs):
        super(Module, self).__init__()
        self.save_hyperparameters(kwargs)

        self.plm_model = plm_model
        self.tokenizer = tokenizer

        self.hidden_size = self.plm_model.config.hidden_size
        self.linear = torch.nn.Linear(self.hidden_size, len(schema2id))

        self.criterion = torch.nn.CrossEntropyLoss()

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

        bio_prob = self.linear(plm_hidden)
        output = {'bio_prob': bio_prob}

        if labels:
            loss_bio = self.criterion(bio_prob.permute(0, 2, 1), labels['bio_seq'])
            output['loss'] = {'bio_loss'  : loss_bio,
                              'total_loss': loss_bio}

        return output

    def predict_step(self, batch, batch_idx: int = 0, dataloader_idx: int = 0):

        outputs = self(**batch)

        # mask special tokens [CLS], [SEP]
        batch_bio_prob = outputs['bio_prob']
        batch_bio_seq = torch.argmax(batch_bio_prob, dim=-1)

        cls_mask = batch.input_ids == self.tokenizer.cls_token_id
        batch_bio_seq[cls_mask] = schema2id['O']

        sep_mask = batch.input_ids == self.tokenizer.sep_token_id
        batch_bio_seq[sep_mask] = schema2id['O']

        return {'bio_seq': batch_bio_seq}

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
        for b in range(len(outputs['bio_seq'])):
            example = {'text': text[b], 'entities': []}

            for t, start, end in decode_bio_seq(outputs['bio_seq'][b], id2schema):
                start_idx, end_idx = int(offset_mapping[b, start, 0]), int(offset_mapping[b, end - 1, 1])

                example['entities'].append({
                        'start_idx': start_idx,
                        'end_idx'  : end_idx - 1,
                        'type'     : t,
                        'entity'   : text[b][start_idx: end_idx]})

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
        json_file_path = os.path.join(dirpath, f'{current_epoch:02d}-{global_step:06d}.json')

        write_json(json_file_path, outputs)
        metrics = eval_CMeEE('data/CMeEE-V2/CMeEE-V2_dev.json', json_file_path)

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
        json_file_path = os.path.join(dirpath, f'test.json')

        write_json(json_file_path, outputs)

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
    train_data = read_json('data/CMeEE-V2/CMeEE-V2_train.json')
    valid_data = read_json('data/CMeEE-V2/CMeEE-V2_dev.json')
    test_data = read_json('data/CMeEE-V2/CMeEE-V2_test.json')

    # init module
    module = Module(plm_model, tokenizer, **vars(module_args))
    dataset = CMeEEDataModule({'train': train_data, 'valid': valid_data, 'test': test_data},
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

    parser = HfArgumentParser([ModuleArguments, CMeEEDataModuleArguments])
    parser = Trainer.add_argparse_args(parser)

    module_args, data_module_args, trainer_args = parser.parse_args_into_dataclasses()

    seed_everything(42, workers=True)
    main(module_args, data_module_args, trainer_args)
