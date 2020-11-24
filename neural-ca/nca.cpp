#include <torch/torch.h>
#include <iostream>

using namespace torch::indexing;

struct NCA : torch::nn::Module {
  NCA () {
  }

  torch::Tensor forward(torch::Tensor input) {
    return torch::zeros({16, 16});
  }

};

int main(int argc, char* argv[]) {
  torch::Tensor world = torch::zeros({16, 16});
  world.index_put_({Slice(), 5}, 1.0);
  world.index_put_({5, Slice()}, 1.0);
  std::cout << world << std::endl;
  std::cout << "Done" << std::endl;
}
