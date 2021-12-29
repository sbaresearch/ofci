# tracer-gen

This provides some script for generating the Tigress dataset. The `tracer` pintool should be built first to execut the traces.

# gen.py

`gen.py` contains the grammar for generating random code files. Check out the code for inner workings, only takes a random seed number as argument and prints generated code to stdout.

# gen-tigress.py

`gen-tigress.py` has been used to generated most of the other `.sh` files. It just creates shellscripts that generate a Tigress cmd containing all functions to be obfuscated, with the same function naming scheme as `gen.py`.

# gen-*.sh

These script files obfuscate all functions generated with `gen.py`, with the corresponding obfuscation category. They take the name of a C-file as input, i.e. `original/test.c` is the input, then the argument should be `test` and the output will be written to `virt/test.c` in the case of `virt`.

# gen-trace-*.sh

These scripts trace the generated virtualized binaries. The paths for Pin and the Pintool are hardcoded and need to be replaced. 
