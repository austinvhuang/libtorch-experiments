#include <stdio.h>
#include <string.h>

int main(void) {
  printf("WASM Ready - with HTML\n");
  return 1;
}

int getNum() { return 22; }

int getDoubleNum(int n) { return n * 2; }

char greeting[50];

char *greet(char *name) {
  if (strlen(name) > 50) {
    return "Name too long";
  }
  strcpy(greeting, "Hello ");
  return strcat(greeting, name);
}