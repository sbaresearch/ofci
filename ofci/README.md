# ofci (Ghidra plugin)

Build using gradle, but need to specify location of your Ghidra installation. The compiled plugin can be found in the `dist` folder as `.zip` file after building; this can be installed through the Ghidra UI or copied to the locations required by the `extractor`.

```
./gradlew -PGHIDRA_INSTALL_DIR=/opt/ghidra build
```
