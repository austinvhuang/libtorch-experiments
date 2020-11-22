
prime:
	emcc lib/prime.c -s WASM=1 -s EXPORTED_FUNCTIONS="['_isPrime', '_checkPrimes']" -o public/prime.js
	# emcc lib/prime.c -s WASM=1 -s EXPORTED_FUNCTIONS="['_isPrime', '_checkPrimes']" -O2 -s TOTAL_MEMORY=64MB  -o public/prime.js