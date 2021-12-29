#!/bin/bash

# From https://stackoverflow.com/questions/59895/how-can-i-get-the-source-directory-of-a-bash-script-from-within-the-script-itsel
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  TARGET="$(readlink "$SOURCE")"
  if [[ $TARGET == /* ]]; then
    SOURCE="$TARGET"
  else
    DIR="$( dirname "$SOURCE" )"
    SOURCE="$DIR/$TARGET" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
  fi
done
RDIR="$( dirname "$SOURCE" )"
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

mkdir -p "/root/extract_logs/$1"

docker run --rm --name $1 \
    --network="dataset_extraction" \
    --mount type=bind,source="/ghidra_extensions",target="/root/.ghidra/.ghidra_10.0.2_PUBLIC/Extensions",readonly \
    --mount type=bind,source="/ofci_ghidra",target=/projects \
    --mount type=bind,source="/root/extract_logs/$1",target=/root/extract_logs \
    -ti ghidra \
    "$1/ghidra_input" \
    -process \
    -readOnly \
    -noanalysis \
    -log "/root/extract_logs/ghidra" \
    -scriptlog "/root/extract_logs/extract" \
    -postScript OFCIExtractorV2.java $1

