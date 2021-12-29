#!/usr/bin/env python3
import os
import sys
from os import walk, path, rename

if __name__ == '__main__':
    dataset_path = sys.argv[1]
    output_dir = sys.argv[2]
    count = 0

    _, dirs, _ = next(walk(dataset_path))
    for d in dirs:
        parts = d.split('-')

        outpath = path.join(output_dir, parts[-1])
        if not path.exists(outpath):
            os.mkdir(outpath)

        p = path.join(dataset_path, d)
        prefix = '-'.join(parts[:-1])
        _, _, filenames = next(walk(p))
        for f in filenames:
            df = f'{prefix}-{f}'
            fp = path.join(p, f)
            dp = path.join(outpath, df)
            os.rename(fp, dp)

            print(dp)
            count += 1

    print(count)
