# Requirements
# cmake - install with homebrew
# new clang - install with homebrew. As of Catalina the system clang++ doesn't support sufficiently recent C++ standards


default: cmake-build

clean:
	rm -r libtorch
	rm -r build
	rm -f libtorch-macos-latest.zip
	rm -f example-app

lib: 
	wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip -O libtorch-macos-latest.zip
	unzip -o libtorch-macos-latest.zip
	rm libtorch-macos-latest.zip

cmake-build: lib
	mkdir -p build
	cd build; cmake -DCMAKE_PREFIX_PATH=$(PWD)/libtorch \
		-DCMAKE_CXX_COMPILER=/usr/local/Cellar/llvm/11.0.0/bin/clang++ ..
	cd build; cmake --build . --config Release
	cd build; ./example-app

scratch-build: lib
	/usr/local/Cellar/llvm/11.0.0/bin/clang++ example-app.cpp \
		-fcolor-diagnostics \
		-I $(PWD)/libtorch/include/torch/csrc/api/include/ \
		-I $(PWD)/libtorch/include \
		-L $(PWD)/libtorch/lib \
		-ltorch_cpu -lc10 \
		-o example-app
	export DYLD_LIBRARY_PATH=$(PWD)/libtorch/lib; ./example-app
