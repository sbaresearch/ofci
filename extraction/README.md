# Data Extraction Pipeline

## Build Docker Image

* Copy postgres jar to this folder

```
docker build . -t ghidra:latest
```

## Run Ghidra against Binaries

```
docker run --rm --name ghidra --mount type=bind,source=/home/anon/projects/obfuscated-fci/src/ofci/lib/build/classes/java/main,target=/ghidra_scripts,readonly --mount type=bind,source=/home/anon/tmp,target=/output --mount type=bind,source=/usr/bin/ls,target=/ghidra_input,readonly -ti ghidra
```
