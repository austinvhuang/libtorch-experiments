#include <torch/torch.h>
#include <iostream>

int main(int argc, char* argv[]) {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
