default: server

# build .wasm and .js 
js:
	emcc demo.c -s WASM=1 -o public/demo.js

# run wasm main() after a javascript file
post-js:
	emcc demo.c -s WASM=1 --post-js public/ready.js -o public/demo.js

# run wasm main() before a javascript file
pre-js:
	emcc demo.c -s WASM=1 --pre-js public/ready.js -o public/demo.js

# optimize file sizes
optimized:
	emcc demo.c -s WASM=1 -O3 -o public/demo.js

# generate html template in addition to .js and .wasm
html:
	emcc demo.c -s WASM=1 -o public/demo.html

# generate .wasm only
wasm:
	emcc demo.c -s WASM=1 -s SIDE_MODULE=1 -o public/demo.wasm

# start express server
server: js
	yarn start

# export c functions to js
# note EXPORTED_FUNCTIONS array is passed as a json string
exports:
	emcc demo.c -s WASM=1 -s EXPORTED_FUNCTIONS="['_getNum', '_getDoubleNum', '_greet', '_main']" -o public/demo.js

imports:
	emcc imports.c -s WASM=1 -o public/imports.js

strings:
	# emcc strings.c -s WASM=1 -s EXPORTED_FUNCTIONS="['_getStr', '_main']" --post-js public/strings-post.js -o public/strings.js
	emcc strings.c -s WASM=1 -s EXPORTED_FUNCTIONS="['_getStr', '_main']" -o public/strings.js

# check by evaluating
# Module.HEAP32.buffer.byteLength
# in the console, should read 64MB instead of default 16MB
memory_buffer:
	emcc strings.c -s WASM=1 -s EXPORTED_FUNCTIONS="['_getStr']" -s TOTAL_MEMORY=64MB -o public/strings.js