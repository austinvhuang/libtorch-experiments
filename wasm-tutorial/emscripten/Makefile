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

# note EXPORTED_FUNCTIONS array is passed as a json string
exports:
	emcc demo.c -s WASM=1 -s EXPORTED_FUNCTIONS="['_getNum', '_getDoubleNum', '_greet', '_main']" -o public/demo.js
