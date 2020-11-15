cmake-build:
	cd pytorch; patch -p1 < ../pytorch.patch
	mkdir -p pytorch/build
	cd pytorch/build; emcmake cmake -DCMAKE_BUILD_TYPE=Release -DUSE_STATIC_DISPATCH=ON -DUSE_NUMA=OFF -DINTERN_BUILD_MOBILE=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_PREFIX_PATH=`pwd`/.. -DPYTHON_EXECUTABLE=/Users/austinhuang/miniconda3/bin/python -DFXDIV_SOURCE_DIR=`pwd`/../third_party/FXdiv -DPSIMD_SOURCE_DIR=`pwd`/../third_party/psimd .. -DCMAKE_CXX_FLAGS=-I/Users/austinhuang/miniconda3/include/glog/
	cd pytorch/build; emcmake make install DESTDIR=.
