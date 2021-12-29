#!/usr/bin/env python3
import argparse
import csv
import sqlite3
import h5py
import numpy as np

# \copy (select f.id, f.name, f.project_group, f.tool, f.category, f.call_count, f.token_count from functions f join function_names fn on fn.id = f.name where fn.name NOT LIKE 'FUN_%') to '/tmp/funcs.csv' with csv
TABLE_CREATION="""CREATE TABLE functions (
    id INT NOT NULL PRIMARY KEY,
    name INT NOT NULL,
    project INT NOT NULL,
    tool INT NOT NULL,
    category INT NOT NULL,
    call_count INT NOT NULL,
    token_count INT NOT NULL,
    emb_id INT NOT NULL
);"""

INSERT_FUNC="""INSERT INTO functions VALUES (?,?,?,?,?,?,?,?);"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Import a database dump of the function table for local generation.'
    )
    parser.add_argument(
        '--database-csv',
        dest='dump',
        type=str,
        help='Path to a csv dump of the function table.',
        required=True
    )
    parser.add_argument(
        '--sqlite-db',
        dest='dbpath',
        type=str,
        help='Output file for the sqlite database',
        required=True
    )
    parser.add_argument(
        '--embeddings',
        dest='emb',
        type=str,
        help='File path to the embeddings hdf5 database.',
        required=True
    )
    parser.add_argument(
        '--blacklist',
        dest='blacklist',
        type=str,
        help='File path to the blacklist of functions used in training.',
        required=True
    )
    args = parser.parse_args()

    # Open db connection and cursor
    con = sqlite3.connect(args.dbpath)

    # Create the table
    with con:
        con.execute(TABLE_CREATION)

    # Load the function blacklist as set
    with open(args.blacklist) as f:
        blacklist = set(f.read().splitlines())
        blacklist_names = {1, 1037, 5596}

    # Load the embedding database
    with h5py.File(args.emb) as f:
        fn_ids = np.array(f['fn_ids'])
        print(f'embed length: {fn_ids.shape[0]}')
        fn_lookup = np.full((np.amax(fn_ids) + 1,), -1, dtype=np.int32)
        fn_lookup[fn_ids] = np.arange(0, fn_ids.shape[0], dtype=np.int32)

    # Parse the exported CSV
    with open(args.dump, newline='') as csvfile:
        funcreader = csv.reader(csvfile, delimiter=',')
        with con:
            for row in funcreader:
                fid         = int(row[0])
                fname       = int(row[1])
                fproject    = int(row[2])
                ftool       = int(row[3])
                fcategory   = int(row[4])
                fcallcount  = int(row[5])
                ftokencount = int(row[6])
        
                # Skip blacklisted functions
                if fproject == 8629:
                    continue
                if f'{fname}-{ftool}' in blacklist or fname in blacklist_names:
                    continue

                # Make sure we have an embedding, otherwise we
                # fucked up somewhere..
                emb = fn_lookup[fid].item()
                assert emb >= 0, f"Missing embedding for {fid}"

                con.execute(INSERT_FUNC, (
                    fid,
                    fname,
                    fproject,
                    ftool,
                    fcategory,
                    fcallcount,
                    ftokencount,
                    emb
                ))

    # Cleanup connection
    con.close()
