# Lightning Information Extraction

Transformers + Linear / CRF / GlobalPointer / EfficientGlobalPointer

## Requirements

```text
python
torch
transformers
pytorch_lightning
```

## Examples

### CLUENER2020

|                                              | val-P   | val-R   | val-F1  | test-F1 |
|----------------------------------------------|---------|---------|---------|---------|
| Mengzi-BERT-L6-H768 + Linear                 | 77.5415 | 77.6197 | 77.5806 | 77.5720 |
| Mengzi-BERT-L6-H768 + CRF                    | 77.3492 | 77.8578 | 77.6027 | 76.8710 |
| Mengzi-BERT-L6-H768 + GlobalPointer          | 79.6403 | 79.3297 | 79.4847 | -       |
| Mengzi-BERT-L6-H768 + EfficientGlobalPointer | 80.4201 | 79.2717 | 79.8418 | -       |
| Mengzi-BERT-Base + Linear                    | 72.3986 | 77.7371 | 74.9729 | -       |
| Mengzi-BERT-Base + CRF                       | 79.2594 | 79.6654 | 79.4619 | -       |
| Mengzi-BERT-Base + GlobalPointer             | 77.4121 | 82.6817 | 79.9602 | 79.3320 |
| Mengzi-BERT-Base + EfficientGlobalPointer    | 79.7005 | 80.7916 | 80.2423 | 79.6910 |

```shell
CUDA_VISIBLE_DEVICES=0 python examples/example_CLUENER_Linear.py --num_workers 8 --device auto --accelerator auto --num_sanity_val_steps 0  --max_epoch 10

CUDA_VISIBLE_DEVICES=0 python examples/example_CLUENER_CRF.py --num_workers 8 --device auto --accelerator auto --num_sanity_val_steps 0  --max_epoch 10

CUDA_VISIBLE_DEVICES=0 python examples/example_CLUENER_GlobalPointer.py --num_workers 8 --device auto --accelerator auto --num_sanity_val_steps 0  --max_epoch 10

CUDA_VISIBLE_DEVICES=0 python examples/example_CLUENER_EfficientGlobalPointer.py --num_workers 8 --device auto --accelerator auto --num_sanity_val_steps 0  --max_epoch 10
```

### CMeEE-V2

|                                              | val-P   | val-R   | val-F1  | test-P  | test-R  | test-F1 |
|----------------------------------------------|---------|---------|---------|---------|---------|---------|
| Mengzi-BERT-L6-H768 + Linear                 | 67.3511 | 61.8764 | 64.4978 | 68.0793 | 62.7880 | 65.3267 |
| Mengzi-BERT-L6-H768 + CRF                    | 68.2281 | 62.6011 | 65.2936 | 68.9237 | 63.5004 | 66.1010 |
| Mengzi-BERT-L6-H768 + GlobalPointer          | 72.9315 | 74.8974 | 73.9014 | 73.4941 | 75.4894 | 74.4784 |
| Mengzi-BERT-L6-H768 + EfficientGlobalPointer | 74.3161 | 73.2333 | 73.7708 | 74.7132 | 73.5768 | 74.1407 |

```shell
CUDA_VISIBLE_DEVICES=0 python examples/example_CMeEE_Linear.py --num_workers 8 --device auto --accelerator auto --num_sanity_val_steps 0  --max_epoch 10

CUDA_VISIBLE_DEVICES=0 python examples/example_CMeEE_CRF.py --num_workers 8 --device auto --accelerator auto --num_sanity_val_steps 0  --max_epoch 10

CUDA_VISIBLE_DEVICES=0 python examples/example_CMeEE_GlobalPointer.py --num_workers 8 --device auto --accelerator auto --num_sanity_val_steps 0  --max_epoch 10

CUDA_VISIBLE_DEVICES=0 python examples/example_CMeEE_EfficientGlobalPointer.py --num_workers 8 --device auto --accelerator auto --num_sanity_val_steps 0  --max_epoch 10
```

### CMeIE

|                                              | val-P   | val-R   | val-F1  | test-P  | test-R  | test-F1 |
|----------------------------------------------|---------|---------|---------|---------|---------|---------|
| Mengzi-BERT-L6-H768 + GlobalPointer          | 69.2548 | 51.9574 | 59.3719 | 68.7590 | 51.2880 | 58.7520 |
| Mengzi-BERT-L6-H768 + EfficientGlobalPointer | 67.2568 | 50.7622 | 57.8569 | 67.0920 | 49.1870 | 56.7610 |

#### RoBERTa + GlobalPointer

```shell
CUDA_VISIBLE_DEVICES=0 python examples/example_CMeIE_GlobalPointer.py --num_workers 8 --device auto --accelerator auto --num_sanity_val_steps 0  --max_epoch 75

CUDA_VISIBLE_DEVICES=0 python examples/example_CMeIE_EfficientGlobalPointer.py --num_workers 8 --device auto --accelerator auto --num_sanity_val_steps 0  --max_epoch 75
```