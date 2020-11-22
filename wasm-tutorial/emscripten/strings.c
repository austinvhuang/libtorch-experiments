#include <emscripten.h>
#include <stdio.h>

char *str = "my string variable";

char *getStr() { return str; }

int main() { return 1; }