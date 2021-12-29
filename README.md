# OFCI

1. Build everything first:
    1. `ofci` with Gradle
    2. `ofci_db` with Cargo
    3. `tracer` with GCC/Pin
    4. Install dependencies for `model` (`requirements.txt`)
2. Build datasets:
    1. Get the [TREX](https://drive.google.com/drive/folders/1FXlrGiZkch9bnAxlrm43IhYGC3r5NveA?usp=sharing) dataset (or use your own with similar naming conventions)
    2. Perform steps mentioned in `extractor`
    3. Get [Tigress](https://tigress.wtf) 3.1 and build a dataset with `tracer-gen`
3. Train (see `model` for details)
4. Embedding generation (see `model` for details)
