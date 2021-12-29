#!/usr/bin/env python3
import argparse
import random
import numpy as np
import torch
import h5py

from transformers import Trainer, TrainingArguments

from ofci.evaluation import AlbertForSequenceSimilarity
from ofci.evaluation import MultiGenerationSet 


if __name__ == '__main__':
    # Chosen by fair dice roll, for reproducability
    torch.manual_seed(2612384816)
    random.seed(5384237557)
    np.random.seed(77954381)

    parser = argparse.ArgumentParser(description='Generate embeddings for a dataset.')
    parser.add_argument(
        '--input',
        dest='input',
        type=str,
        help='Path to a h5 inference dataset.',
        required=True
    )
    parser.add_argument(
        '--model',
        dest='model',
        type=str,
        help='Path to a finetuned OFCI model.',
        required=True
    )
    parser.add_argument(
        '--output-dir',
        dest='outdir',
        type=str,
        help='Output directory for h5 datasets with final embeddings',
        required=True
    )
    parser.add_argument(
        '--cutoff',
        dest='cutoff',
        type=int,
        help='Maximum number of entries to be used from the dataset'
    )
    args = parser.parse_args()

    # Check if CUDA is available
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    # Setup Albert
    model = AlbertForSequenceSimilarity.from_pretrained(args.model)
    print(f'Model parameters: {model.num_parameters():_}')

    # Load the finetuning data from file
    dataset = MultiGenerationSet(args.input, cutoff=args.cutoff)
    print(f'Number of dataset splits: {len(dataset)}')

    # Init trainer
    training_args = TrainingArguments(
        output_dir=args.outdir,
        overwrite_output_dir=False,
        per_device_eval_batch_size=128,
        eval_accumulation_steps=100,
        dataloader_num_workers=4,
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=training_args)
    print('Trainer initialized.')

    # Prediction loop
    for i in range(len(dataset)):
        chunk_id = f'{i:03}'
        print(f'Predicting chunk {chunk_id}...')
        ds = dataset[i]

        p = trainer.predict(ds)
        print(p.metrics)

        with h5py.File(f'{args.outdir}/emb-{chunk_id}', 'w') as f:
            print(f'Writing chunk {chunk_id}...')
            f.create_dataset('embeddings', data=p.predictions)

        print(f'Done writing {chunk_id}...')
