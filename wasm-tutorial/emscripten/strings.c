#include <emscripten.h>
#include <stdio.h>

char *str = "my string variable";

int getNumber() {
  int num = 22;
  // emscripten_debugger(); // trigger breakpoint
  if (num < 30) {
    emscripten_log(EM_LOG_WARN, "num less than 30: %d", num);
  }
  return num;
}

char *getStr() { return str; }

int main() { return 1; }