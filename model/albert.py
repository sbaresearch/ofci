#!/usr/bin/env python3
import torch
import h5py
import numpy as np

from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# MODEL SPECIFIC
from transformers import AlbertConfig 
from transformers import AlbertForMaskedLM
from transformers import RobertaTokenizerFast


VOCAB_DIR = '/experiments/vocab'
OUTPUT_DIR = './training'
DATASET = '/experiments/pretrain-full.h5'
PADDING_TOKEN = 1


class OFCIPretrainDataset(Dataset):
  def __init__(self, data_tensor):
    self.elements = data_tensor
    self.attention_mask = (data_tensor != PADDING_TOKEN).float()

  def __len__(self):
    return self.elements.size()[0]

  def __getitem__(self, idx):
    return {
        'input_ids': self.elements[idx],
        'attention_mask': self.attention_mask[idx],
        'labels': self.elements[idx]
    }


def load_data(file):
    f = h5py.File(file)
    idx = torch.randperm(f['data'].shape[0])
    print('start loading dataset...')
    data_shuffled = torch.tensor(np.array(f['data']))[idx][:1_000_000].long()
    print('finished loading dataset.')
    f.close()
    return data_shuffled[:10_000], data_shuffled[10_000:]


if __name__ == '__main__':
    print(torch.cuda.is_available())

    # Load tokenizer
    tokenizer = RobertaTokenizerFast(
        vocab_file=f'{VOCAB_DIR}/vocab.json',
        merges_file=f'{VOCAB_DIR}/merges.txt'
    )
    vocab_size = len(tokenizer.get_vocab())
    print(f'vocab_size: {vocab_size:_}')

    config = AlbertConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        intermediate_size=3072,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        save_steps=10_000,
        save_total_limit=10,
        prediction_loss_only=False,
        fp16=True,
        dataloader_num_workers=4,
        #fp16_full_eval=True,
        do_eval=True,
        do_train=True,
        evaluation_strategy='epoch'
    )

    # Setup Roberta
    model = AlbertForMaskedLM(config=config)
    print(f'Model parameters: {model.num_parameters():_}')

    # Load all the data from file
    data_valid, data_train = load_data(DATASET)
    print(f'Training set size: {data_train.size()[0]:_}')

    # Init dataset
    trainset = OFCIPretrainDataset(data_train)
    validset = OFCIPretrainDataset(data_valid)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.2,
    )

    # Init trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=trainset,
        eval_dataset=validset
    )

    print('Trainer initialized.')

    train_result = trainer.train()
    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(trainset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    metrics = trainer.evaluate(eval_dataset=validset)
    metrics["eval_samples"] = len(validset) 
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

