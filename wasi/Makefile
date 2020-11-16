wasi-sdk:
	wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-11/wasi-sdk-11.0-macos.tar.gz
	tar -xzf wasi-sdk-11.0-macos.tar.gz
	wasi-sdk-11.0/bin/clang++ --sysroot=./wasi-sdk-11.0/share/wasi-sysroot -Iwasi-sdk-11.0/share/wasi-sysroot/include -Iwasi-sdk-11.0/share/wasi-sysroot/include/c++/v1 hello.cpp -o hello.wasm
	wasmtime hello.wasm
	rm wasi-sdk-11.0-macos.tar.gz
