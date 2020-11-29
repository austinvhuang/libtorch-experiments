default: cmake-build

clean:
	rm -r libtorch
	rm -r build
	rm -f libtorch-macos-latest.zip
	rm -f nca

lib: 
	wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip -O libtorch-macos-latest.zip
	unzip -o libtorch-macos-latest.zip
	rm libtorch-macos-latest.zip

cmake-build: lib
	mkdir -p build
	cd build; cmake -DCMAKE_PREFIX_PATH=$(PWD)/libtorch \
		-DCMAKE_CXX_COMPILER=/usr/local/Cellar/llvm/11.0.0/bin/clang++ ..
	cd build; cmake --build . --config Release
	cd build; ./nca

scratch-build: lib
	/usr/local/Cellar/llvm/11.0.0/bin/clang++ nca.cpp \
		-fcolor-diagnostics \
		-I $(PWD)/libtorch/include/torch/csrc/api/include/ \
		-I $(PWD)/libtorch/include \
		-L $(PWD)/libtorch/lib \
		-ltorch_cpu -lc10 \
		-std=c++17
		-o nca
	export DYLD_LIBRARY_PATH=$(PWD)/libtorch/lib; ./nca

rebuild1:
	mkdir -p build
	cd build; cmake -DCMAKE_PREFIX_PATH=$(PWD)/libtorch \
		-DCMAKE_CXX_FLAGS="-g -std=c++17" \
		-DCMAKE_CXX_COMPILER=/usr/local/Cellar/llvm/11.0.0/bin/clang++ ..
	cd build; cmake --build . --config Debug
	cd build; ./nca

rebuild2:
	time /usr/local/Cellar/llvm/11.0.0/bin/clang++ nca.cpp \
		-fcolor-diagnostics \
		-I $(PWD)/libtorch/include/torch/csrc/api/include/ \
		-I $(PWD)/libtorch/include \
		-L $(PWD)/libtorch/lib \
		-ltorch_cpu -lc10 \
		-o nca
	export DYLD_LIBRARY_PATH=$(PWD)/libtorch/lib; ./nca

em-nca: 
	emcc nca.cpp \
		-fcolor-diagnostics \
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
		-o nca.html


watcher:
		rg nca.cpp Makefile --files | entr -s "make rebuild1"
