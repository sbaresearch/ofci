import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from torch import nn
from torch.utils.data import Dataset
from transformers import AlbertModel
from transformers import AlbertPreTrainedModel


PADDING_TOKEN=1


class AlbertSimilarityHead(nn.Module):
    """Head for sequence similarity tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) 
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
        input_ids=None,
        labels=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ): 
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output = outputs[0]
        return self.similarity(sequence_output)


class GenerationDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask 

    def __len__(self):
        return self.input_ids.size()[0]

    def __getitem__(self, idx):
        data = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }
        return data

class MultiGenerationSet(Dataset):
    def __init__(self, filepath, cutoff=None, split=100_000):
        with h5py.File(filepath) as f:
            print(f'Loading {filepath}')
            if cutoff:
                input_ids = torch.tensor(np.array(f['fragments']))[:cutoff].long()
            else:
                input_ids = torch.tensor(np.array(f['fragments'])).long()
            print(f'Finished loading {filepath}')

        # Inputs should have shape (N, 512)
        self.input_ids = torch.split(input_ids, split)
        self.attention_mask = torch.split((input_ids != PADDING_TOKEN).float(), split)
        self.datasets = [
            GenerationDataset(self.input_ids[i], self.attention_mask[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, i):
        return self.datasets[i]

