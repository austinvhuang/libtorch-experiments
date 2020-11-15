# Notes on wasm application building

See https://github.com/junjihashimoto/pytorch-wasm particularly the workflow process https://github.com/junjihashimoto/pytorch-wasm/blob/master/.github/workflows/wasm.yml

## Misc

See https://github.com/pytorch/pytorch/issues/16112 - install:
- glog
- yaml
```
/Users/austinhuang/libtorch-experiments/wasm/pytorch/build/CMakeFiles/CMakeTmp/src.cxx:1:10: fatal error: 'glog/stl_logging.h' file not found
#include <glog/stl_logging.h>
         ^~~~~~~~~~~~~~~~~~~~
```
