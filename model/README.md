# model

This contains the implementation of the Albert model as used in Ofci and the training process.

# Pre-Training

Use `albert.py` for pre-training. It contains several hardcoded paths as variables that need to be adjusted.

# Fine-Tuning

Use `finetune-albert.py` for fine-tuning. It contains several hardcoded paths as variables that need to be adjusted.

# Embedding Generation

The embedding generation is performed over several steps. Every script requires their own command-line parameters, which are documented through the python standard command line arg parsing system.

1. To start off, the inference database has to be generated with `ofci_db`, which will produce the raw data needed for embedding generation.
2. `evaluate.py` will generate the raw embeddings for the inference dataset.
3. Because the embedding database contains just fragments, it still needs to adjust for the case when functions contain multiple fragments. First, `fragment-mean-indices.py` needs to be called in order to generate the indices to keep track off which fragments to combine.
4. `fragment-calculate-means.py` will take the indices and calculate the embedding mean for functions that consist of more than one fragment, and outputs the final embeddings.
5. To combine all of this together with the metadata from the database, `sqlite-importer.py` takes a CSV dump of the extraction database and combines it into an sqlite database that allows direct mapping of metadata to offsets in the embedding file.
