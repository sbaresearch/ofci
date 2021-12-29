#!/usr/bin/env python3
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from typing import Optional
from torch import nn
from torch.utils.data import Dataset
from transformers.file_utils import ModelOutput
from transformers import Trainer, TrainingArguments
from transformers import EvalPrediction

# MODEL SPECIFIC
from transformers import AlbertConfig
from transformers import AlbertModel
from transformers import AlbertPreTrainedModel
from transformers import RobertaTokenizerFast


VOCAB_DIR = '/data/vocab'
OUTPUT_DIR = '/data/finetune-coreutils-old-pretrain-old'
MODEL_PATH = '/data/albert-pretrain-old'
DATASET = '/data/finetune-coreutils-old.h5'
PADDING_TOKEN = 1


@dataclass
class SequenceSimilarityOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits0: Optional[torch.FloatTensor] = None
    logits1: Optional[torch.FloatTensor] = None


class AlbertSimilarityHead(nn.Module):
    """Head for sequence similarity tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) 
        # TODO: quant noise/spectral norm? 
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, features):
        x = torch.mean(features, dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = F.normalize(x)
        return x


class AlbertForSequenceSimilarity(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.albert = AlbertModel(config)
        self.similarity = AlbertSimilarityHead(config)
        self.init_weights()

    def forward(
        self,
        input0: torch.LongTensor = None,
        input1: torch.LongTensor = None,
        attention_mask0: torch.LongTensor = None,
        attention_mask1: torch.LongTensor = None,
        token_type_ids=None,
        position_ids=None,
        labels: torch.FloatTensor = None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ) -> SequenceSimilarityOutput:
        # Feed first function fragment to model
        outputs = self.albert(
            input0,
            attention_mask=attention_mask0,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output = outputs[0]
        logits0 = self.similarity(sequence_output)

        # Feed second function fragment to model
        outputs = self.albert(
            input1,
            attention_mask=attention_mask1,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output = outputs[0]
        logits1 = self.similarity(sequence_output)

        # Calculate cosine similarity loss
        loss = F.cosine_embedding_loss(
            logits0,
            logits1,
            labels,
            margin=self.config.cosine_embedding_loss_margin,
            reduction='sum', # TODO: sum vs mean?
        )

        # TODO: Could try cosine similarity + MSE loss
        # TODO: logging?

        return SequenceSimilarityOutput(
            loss=loss,
            logits0=logits0,
            logits1=logits1
        )


class OFCISimilarityDataset(Dataset):
    def __init__(
        self,
        input0: torch.LongTensor,
        input1: torch.LongTensor,
        labels: torch.FloatTensor
    ):
        # Inputs should have shape (N, 512)
        # Targets should have shape (N)
        self.input0 = input0
        self.attention_mask0 = (input0 != PADDING_TOKEN).float()
        self.input1 = input1
        self.attention_mask1 = (input1 != PADDING_TOKEN).float()
        self.labels = labels

    def __len__(self):
        return self.labels.size()[0]

    def __getitem__(self, idx):
        data = {
            'input0': self.input0[idx],
            'attention_mask0': self.attention_mask0[idx],
            'input1': self.input1[idx],
            'attention_mask1': self.attention_mask1[idx],
            'labels': self.labels[idx],
        }
        return data


def load_data(file):
    f = h5py.File(file)
    print('start loading dataset...')
    fn1 = torch.tensor(np.array(f['fn1'])).long()
    fn2 = torch.tensor(np.array(f['fn2'])).long()
    labels = torch.tensor(np.array(f['labels'])).float()
    print('finished loading dataset.')
    f.close()
    return fn1, fn2, labels 


if __name__ == '__main__':
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    # Load Tokenizer
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
        num_hidden_layers=4,
        type_vocab_size=1,
        cosine_embedding_loss_margin=0.1,
    )
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=30,
        per_device_train_batch_size=18,
        per_device_eval_batch_size=18,
        save_strategy='epoch',
        save_steps=50,
        save_total_limit=20,
        prediction_loss_only=False,
        #fp16=True,
        dataloader_num_workers=2,
        #fp16_full_eval=True,
        do_eval=True,
        do_train=True,
        evaluation_strategy='epoch',
        gradient_accumulation_steps=29
    )

    # Setup Albert
    model = AlbertForSequenceSimilarity.from_pretrained(
        MODEL_PATH,
        config=config,
    )
    print(f'Model parameters: {model.num_parameters():_}')

    # Load the finetuning data from file
    data_fn1, data_fn2, data_labels = load_data(DATASET)

    # Initialize the datasets
    trainset = OFCISimilarityDataset(
        input0=data_fn1[:10_000],
        input1=data_fn2[:10_000],
        labels=data_labels[:10_000]
    )

    # Prepare validation set
    idx = torch.randperm(data_fn1[10_000:].shape[0])
    validset = OFCISimilarityDataset(
        input0=data_fn1[10_000:][idx][:10_000],
        input1=data_fn2[10_000:][idx][:10_000],
        labels=data_labels[10_000:][idx][:10_000]
    )

    # Setup metrics computation
    def compute_metrics(p: EvalPrediction):
        logits0 = torch.tensor(p.predictions[0])
        logits1 = torch.tensor(p.predictions[1])
        preds = torch.cosine_similarity(logits0, logits1)
        labels = torch.tensor(p.label_ids)

        # TREX metrics
        ncorrect = (((preds > config.cosine_embedding_loss_margin) ==
            (labels > config.cosine_embedding_loss_margin)) * 
            (labels > config.cosine_embedding_loss_margin)).sum().item()
        ncorrect_total = ((preds > config.cosine_embedding_loss_margin) ==
            (labels > config.cosine_embedding_loss_margin)).sum().item()
        ncorrect_pred = (preds > config.cosine_embedding_loss_margin).sum().item()
        ncorrect_actual = (labels > config.cosine_embedding_loss_margin).sum().item()

        # ROCAUC
        rocauc = roc_auc_score((labels > 0), preds)

        return {
            'ncorrect': ncorrect,
            'ncorrect_total': ncorrect_total,
            'ncorrect_pred': ncorrect_pred,
            'ncorrect_actual': ncorrect_actual,
            'rocauc': rocauc
        }

    # Init trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=validset,
        compute_metrics=compute_metrics,
    )

    print('Trainer initialized.')

    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(trainset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate(eval_dataset=validset)
    metrics["eval_samples"] = len(validset) 
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

