# Notes on building a libtorch app on MacOSX (Catalina)

Run `make`, see `Makefile` for details.

# Troubleshooting

1. Make sure cmake is available. Install cmake with homebrew using `brew install cmake`.

2. Libtorch uses modern C++ not supported by the system clang compiler. Install clang via homebrew using `brew install llvm` and make sure the brew version is used by the build system by passing `-DCMAKE_CXX_COMPILER=[path to homebrew clang]` as an argument to `cmake`.

3. Make sure to path the path to the libtorch directory to cmake - `-DCMAKE_PREFIX_PATH=$(PWD)/libtorch`.

3. Make sure to use the macos libtorch binaries and not the linux ones. macosx binaries are available at: https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip 

4. When building without linking (see `scratch-build`) make sure the shared libraries (`.dylib` files) are referenced in the `DYLD_LIBRARY_PATH` variable.
