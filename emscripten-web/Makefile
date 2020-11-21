default: run

clean:
	rm -f example-app.js
	rm -f example-app.wasm

# run this once before build steps
patch-pytorch:
	cd pytorch; patch -p1 < ../pytorch.patch

cmake-build:
	mkdir -p pytorch/build
	cd pytorch/build; \
		emcmake cmake \
		-DCMAKE_BUILD_TYPE=Release \
		-DUSE_STATIC_DISPATCH=ON \
		-DUSE_NUMA=OFF \
		-DINTERN_BUILD_MOBILE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DCMAKE_PREFIX_PATH=`pwd`/.. \
		-DPYTHON_EXECUTABLE=/Users/austinhuang/miniconda3/bin/python \
		-DFXDIV_SOURCE_DIR=`pwd`/../third_party/FXdiv \
		-DPSIMD_SOURCE_DIR=`pwd`/../third_party/psimd \
		-DCMAKE_CXX_FLAGS=-I/Users/austinhuang/miniconda3/include/glog/ -s STANDALONE_WASM -s DISABLE_EXCEPTION_CATCHING=0 ERROR_ON_UNDEFINED_SYMBOLS=0 -s ALLOW_MEMORY_GROWTH=1 \
		..
	cd pytorch/build; emcmake make install DESTDIR=.

example:
	emcc example-app.cpp \
 		-Lpytorch/build/usr/local/lib \
 		-lclog -lcpuinfo \
		-Wl,--whole-archive \
 		-ltorch_cpu -ltorch -lc10 \
		-Wl,--no-whole-archive \
 		-Ipytorch/build/usr/local/include \
 		-Ipytorch/build/usr/local/include/torch/csrc/ \
 		-Ipytorch/build/usr/local/include/torch/csrc/api/include/ \
 		-Ibuild/usr/local/include/torch/csrc/api/include \
		-s DISABLE_EXCEPTION_CATCHING=0 -Wno-parentheses \
 		pytorch/third_party/cpuinfo/src/emscripten/init.c \
 		-Ipytorch/third_party/cpuinfo/src \
		-s ERROR_ON_UNDEFINED_SYMBOLS=0 \
		-s ALLOW_MEMORY_GROWTH=1 \
 		-o example-app.js
 
run: example
	node example-app.js

# run this before wasi
wasmtime: 
	curl https://wasmtime.dev/install.sh -sSf | bash

hellowasi:
	emcc hello.cpp -o hello.wasm
	wasmtime hello.wasm
