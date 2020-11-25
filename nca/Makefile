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
		-o nca
	export DYLD_LIBRARY_PATH=$(PWD)/libtorch/lib; ./nca

rebuild1:
	mkdir -p build
	cd build; cmake -DCMAKE_PREFIX_PATH=$(PWD)/libtorch \
		-DCMAKE_CXX_FLAGS="-g" \
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

watcher:
		rg nca.cpp Makefile --files | entr -s "make rebuild1"
