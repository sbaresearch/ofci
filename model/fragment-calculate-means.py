#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import torch


EMBEDDING_DIM=768


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate mean embeddings for multi-fragment functions.'
    )
    parser.add_argument(
        '--fragment-info',
        dest='fraginfo',
        type=str,
        help='Path to a h5 fragment mean info dataset.',
        required=True
    )
    parser.add_argument(
        '--fragment-embedding-dir',
        dest='embdir',
        type=str,
        help='Path to a directory containing h5 fragment embedding datasets.',
        required=True
    )
    parser.add_argument(
        '--output-dir',
        dest='outdir',
        type=str,
        help='Output directory for h5 datasets with final embeddings',
        required=True
    )
    args = parser.parse_args()

    # Open fragment info for the dataset
    with h5py.File(args.fraginfo) as frag_info:
        # Estimate size of dataset
        max_fns = sum([c.shape[0] for c in frag_info.values()]) 

        # Preallocate output data
        embeddings = torch.empty((max_fns, EMBEDDING_DIM), dtype=torch.float32)
        fn_ids = torch.empty((max_fns,), dtype=torch.int32)
        fn_count = 0
        last_fn_data = None

        # Iterate over all embedding outputs
        # NOTE: It is necessary that .keys() returns the keys in
        #       the correct order, verify output of script just
        #       to be sure
        for chunk_name in frag_info.keys():
            print(f'Processing "{chunk_name}"...')

            with h5py.File(f'{args.embdir}/{chunk_name}') as f:
                # Load the whole chunk of fragment embeddings
                chunk_emb = torch.tensor(np.array(f['embeddings']))
                chunk_info = torch.tensor(np.array(frag_info[chunk_name]))

                # Iterate over all chunk info and calculate the means
                for i in range(chunk_info.size()[0]):
                    frag_fun   = chunk_info[i][0]
                    frag_start = chunk_info[i][1]
                    frag_end   = chunk_info[i][2]
                    frag_slice = chunk_emb[frag_start:frag_end]

                    # Check if we crossed chunk boundaries and the last
                    # function is equal to the current
                    if last_fn_data and last_fn_data[0] == frag_fun:
                        print(f'found duplicate fn {fn_count - 1}, {frag_fun}, combining embeddings')
                        frag_slice = torch.cat((last_fn_data[1], frag_slice))
                        store_index = fn_count - 1
                    else:
                        store_index = fn_count

                    # Store function for later
                    last_fn_data = (frag_fun, frag_slice)

                    # Calculating embedding mean
                    emb = torch.mean(frag_slice, dim=0)
                    assert emb.size()[0] == EMBEDDING_DIM, "wrong embedding dimension"

                    # Store final embedding
                    embeddings[store_index] = emb
                    fn_ids[store_index] = frag_fun
                    fn_count = store_index + 1
        
    # Alright, by now we have calculated all embedding means
    # Now trim embeddings to actual count and create the final dataset
    print(f'Number of function embeddings: {fn_count}')
    embeddings = embeddings[:fn_count]
    fn_ids = fn_ids[:fn_count]

    print('Writing embedding output file...')
    with h5py.File(f'{args.outdir}/embeddings.h5', 'w') as f:
        f.create_dataset('embeddings', data=embeddings.numpy())
        f.create_dataset('fn_ids', data=fn_ids.numpy())
    print('Finished embedding averaging.')

