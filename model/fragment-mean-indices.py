#!/usr/bin/env python3
import argparse
import numpy as np
import h5py


CHUNK_BOUNDARY = 100_000


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate indices for calculating fragment embedding means.'
    )
    parser.add_argument(
        '--input',
        dest='input',
        type=str,
        help='Path to a h5 inference dataset.',
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

    # Load IDs from dataset
    with h5py.File(args.input) as f:
        fn_ids = np.array(f['ids'])

    # Preallocate output data
    index_data = np.empty(shape=(fn_ids.shape[0], 3), dtype=np.int32)

    # Requires at least one item to be present
    last_id = fn_ids[0]
    last_index = 0
    last_chunk_index = 0
    chunks = []
    unique_count = 0

    # Basically find the index range for all function fragments so we
    # can calculate the embedding mean
    for i in range(fn_ids.shape[0]): 
        current_id = fn_ids[i]

        if last_id != fn_ids[i]:
            last_index_chunk = last_index // CHUNK_BOUNDARY
            i_chunk = i // CHUNK_BOUNDARY

            # Special case if a block is on chunk borders
            if i_chunk != last_index_chunk:
                index_data[unique_count][0] = last_id
                index_data[unique_count][1] = last_index % CHUNK_BOUNDARY
                index_data[unique_count][2] = CHUNK_BOUNDARY
                unique_count += 1

                # Add chunks to list
                chunks.append((last_chunk_index, unique_count))
                last_chunk_index = unique_count

                # If the block is split across chunks, add the second part
                i_mod = i % CHUNK_BOUNDARY
                if i_mod > 0:
                    index_data[unique_count][0] = last_id
                    index_data[unique_count][1] = 0
                    index_data[unique_count][2] = i_mod
                    unique_count += 1
            else:
                # Store info in array
                index_data[unique_count][0] = last_id
                index_data[unique_count][1] = last_index % CHUNK_BOUNDARY
                index_data[unique_count][2] = i % CHUNK_BOUNDARY
                unique_count += 1

            # Update loop vars
            last_id = current_id
            last_index = i

    # Store info of last entry
    # NOTE: This does not handle the special case in which the
    #       last function has more fragments than a chunk can
    #       contain
    index_data[unique_count][0] = last_id
    index_data[unique_count][1] = last_index % CHUNK_BOUNDARY
    index_data[unique_count][2] = fn_ids.shape[0] % CHUNK_BOUNDARY
    if index_data[unique_count][2] == 0:
        index_data[unique_count][2] = CHUNK_BOUNDARY
    unique_count += 1

    # Store info of last chunk
    chunks.append((last_chunk_index, unique_count))
    last_chunk_index = unique_count

    # Trim data array
    index_data = index_data[:unique_count]

    # Store chunk info in h5py file
    with h5py.File(f'{args.outdir}/chunk-embedding-index.h5', 'w') as f:
        for (i, (a,b)) in enumerate(chunks):
            f.create_dataset(f'emb-{i:03}', data=index_data[a:b])

