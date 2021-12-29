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

docker run --rm --name $2 \
    --network="dataset_extraction" \
    --mount type=bind,source="/ghidra_extensions",target="/root/.ghidra/.ghidra_10.0.2_PUBLIC/Extensions",readonly \
    --mount type=bind,source="$1",target=/ghidra_input,readonly \
    --mount type=bind,source="/ofci_ghidra",target=/projects \
    -ti ghidra \
    $2 \
    -import /ghidra_input \
    -preScript OFCIAnalysisSelection.java \
    -postScript ResolveX86orX64LinuxSyscallsScript.java

