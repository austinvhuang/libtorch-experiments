#include <emscripten.h>
#include <stdio.h>

// declare a reusable JS function
EM_JS(void, jsFunction, (int n), { console.log("Call from EM_JS: " + n); });

int main(void) {
  printf("WASM Ready\n");

  // call js function
  emscripten_run_script("console.log('Hello from C!')");
  emscripten_async_run_script("console.log('Hello from C - ASYNC!')",
                              2000); // 2000 = time in ms

  int jsVal = emscripten_run_script_int("getNum()");
  char *jsValStr = emscripten_run_script_string("getStr()");

  printf("value from getNum() called in javascript: %d\n", jsVal);
  printf("value from getStr() called in javascript: %s\n", jsValStr);

  jsFunction(144);

  return 1;
}