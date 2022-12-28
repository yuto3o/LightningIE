# -*- coding: utf-8 -*-
import json
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SequenceDataset(Dataset):

    def __init__(self, data):
        self.data_source = data

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, i):
        return self.data_source[i]


# def decode_bio_seq(seq: torch.LongTensor, id2schema: dict):
#     """ decode sequence (type, start, end) from BIO tag prob
#     Args:
#         seq: torch.LongTensor, (L,)
#         id2schema: dict
#
#     Returns:
#         List[str, int, int]: type, start, end
#     """
#     spans = []
#     start = False
#
#     for i, token_i in enumerate(seq):
#         token = id2schema[token_i]
#
#         if token != 'O':
#
#             if token.startswith('B-'):
#
#                 start = True
#                 spans.append([token.split('-')[1], i, i + 1])
#
#             elif start:
#                 spans[-1][2] = i + 1
#
#             else:
#                 start = False
#         else:
#             start = False
#
#     return spans


def decode_bio_seq(seq: torch.LongTensor, id2schema: dict):
    """ decode sequence (type, start, end) from BIO tag prob
    Args:
        seq: torch.LongTensor, (L,)
        id2schema: dict

    Returns:
        List[str, int, int]: type, start, end
    """
    seq_decode = []
    chunk = [-1, -1, -1]

    for i, token_i in enumerate(seq):
        token = id2schema[token_i]
        if token.startswith('B-'):

            # early stop
            if chunk[2] != -1:
                seq_decode.append(chunk)

            chunk = [-1, -1, -1]
            chunk[0] = token.split('-')[1]
            chunk[1] = i
            chunk[2] = i + 1

            if i == len(seq) - 1 and chunk[2] != -1:
                seq_decode.append(chunk)

        elif token.startswith('I-') and chunk[1] != -1:
            t = token.split('-')[1]

            if t == chunk[0]:
                chunk[2] = i + 1

            if i == len(seq) - 1 and chunk[2] != -1:
                seq_decode.append(chunk)

        else:
            if chunk[2] != -1:
                seq_decode.append(chunk)

            chunk = [-1, -1, -1]

    return seq_decode


def apply_RoPE_position_embeddings(sinusoidal, *tensors) -> List[torch.Tensor]:
    """ apply RoPE position to tensors
        code from https://github.com/bojone/bert4keras/blob/master/bert4keras/backend.py
    Args:
        sinusoidal: torch.Tensor
        *tensors: List[torch.Tensor]

    Returns:
        List[torch.Tensor]
    """
    ndim = tensors[0].ndim
    sinusoidal = align(sinusoidal, [0, 1, -1], ndim)
    cos_pos = torch.repeat_interleave(sinusoidal[..., 1::2], 2, -1)
    sin_pos = torch.repeat_interleave(sinusoidal[..., ::2], 2, -1)

    outputs = []
    for tensor in tensors:
        tensor2 = torch.stack([-tensor[..., 1::2], tensor[..., ::2]], ndim)
        tensor2 = torch.reshape(tensor2, tensor.shape)
        outputs.append(tensor * cos_pos + tensor2 * sin_pos)

    return outputs[0] if len(outputs) == 1 else outputs


def align(tensor, axes, ndim=None):
    ndim = ndim or max(axes) + 1
    indices = [None] * ndim
    for i in axes:
        indices[i] = slice(None)
    return tensor[indices]


def sequence_padding(data: List[torch.Tensor], max_length: int = None, dim: int = -1,
                     value: int = -100) -> torch.Tensor:
    """ automatic sequence padding at 'dim'
    Args:
        data: List[torch.Tensor]
        max_length: int
        value: int
        dim: int

    Returns:
        torch.Tensor
    """
    dim = data[0].ndim + dim if dim < 0 else dim - 1
    if max_length is None:
        max_length = 1
        for _ in data:
            max_length = max(_.size(dim), max_length)

    output = []
    for _ in data:
        pad = []
        for i in range(_.ndim, 0, -1):
            if i - 1 == dim:
                pad += [0, max_length - _.size(dim)]
            else:
                pad += [0, 0]

        output.append(F.pad(_, pad, 'constant', value))

    return torch.stack(output, dim=0)


def read_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return json.load(file)


def write_json(file_name, obj):
    with open(file_name, 'w', encoding='utf-8') as file:
        s = json.dumps(obj, ensure_ascii=False)
        file.write(s + '\n')


def read_jsonl(file_name):
    output = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            output.append(json.loads(line))
    return output


def write_jsonl(file_name, obj):
    with open(file_name, 'w', encoding='utf-8') as file:
        for _ in obj:
            s = json.dumps(_, ensure_ascii=False)
            file.write(s + '\n')
