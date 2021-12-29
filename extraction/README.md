# extractor (Data Extraction Pipeline)

The extraction process requires a few different steps and ties in with `ofci_db`.

## Setting up the TREX dataset

The TREX dataset is structured in a different way than what we want. We want all of the data split up into folders according to category (`O0`, `O1`, `bcf`, `cff`, ...) and the dataset has to be prepared accordingly. Additionally, the files have to roughly correspond to the format of `coreutils-8.32-ls`, i.e. `group-version-tool`, which is being used by the Ghidra extractor to fill database information. Also, Ghidra headless cannot easily deal with static libraries, meaning all static libraries have to be extracted into object files before passing them to the Ghidra extractor.

## Starting a database server

Any (recent) PostgreSQL server can be used, we used the docker-compose setup as provided in the `deploy` folder. Simply running `docker-compose up -d` should do the trick. Afterwards, the database has to be initialized through the migrations described in `ofci_db`.

## Build extractor Docker image

The following will build the Ghidra image required for the extractor to work. The Ghidra version can be changed, but needs to be the same the plugin from `ofci` was compiled with.

```
docker build . -t ghidra:latest
```

## Generate Ghidra project files

In general, extraction and Ghidra project file creation can be combined, for testing purposes we split the process however. During project file creation, the initial binary analysis is performed, which can take quite some time. Project file creation is handled in `create_projects.sh`; For it to function, the paths of the current system have to be adjusted, namely `/ghidra_extensions` to the folder where the (extracted) Ghidra plugin is located (mirroring the extensions folder of an actual Ghidra installation) and `/ofci_ghidra`, the location where the project files are stored. If this is done, `create_projects.sh` can be called with the path to the directory where binary files are located as first argument and a category name as second argument. 

## Extract OFCI data from Ghidra project files

For the Ghidra export functionality, `extract.sh` is used, mostly mirroring `create_projects.sh`. In addition to the previously mentioned paths, `/root/extract_logs` has to be adjusted to a directory for export log storage. Also, the extract script does not take a path argument, but only the category name argument previously provided to `create_projects.sh`.

## Extract OFCI trace data from Ghidra project files

Same as `extract.sh`, but an additional path is required to specify the location of the trace files (`/ofci/$1-traces`). For generating the traces for our dataset, check `tracer-gen`. 
