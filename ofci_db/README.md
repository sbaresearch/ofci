# ofci_db

Rust implementation for database/embedding handling.

## Setup Database

Before the extraction tools can be used, the extraction database has to be initialized. Database migrations are handled with `diesel`, make sure to edit the `.env` file in this directory and then run the following:

```
cargo install diesel_cli
diesel migration run
```

## Build the DB tools

Building the DB tools should be simple, has been tested with Rust 1.56.1:

```
cargo build --release
```

## Dataset generation steps

In order to generate the needed datasets, the DB tools are split up into different binaries, which make use of the database defined in `.env` to connect to the database. These binaries can be found in the cargo output directory or executed with, e.g. `cargo run --release --bin ofci-genvocab`.

1. **ofci-genvocab**: Takes a path and an output file prefix as arguments. Will read all the tokens in the database and generate a BPE vocabulary.
2. **ofci-tokenize**: Takes the path to the vocabulary json and the path to the merges.txt file generated in the previous step and performs tokenization on all data in the database.
3. **ofci-generate-pretrain**: Generates the database for pretraining, takes an output path as argument
4. **ofci-generate-finetune**: Generates the database for finetuning, takes an output path as argument. Also produces a blacklist file for later excluding used functions from evaluation.
5. **ofci-generate-inference**: Dumps all functions for embedding inference, takes an output path as argument and requires the blacklist file to be present in the output directory.
