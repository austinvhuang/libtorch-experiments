#include <array>
#include <iostream>
#include <torch/torch.h>

using namespace torch::indexing;

struct NCA : torch::nn::Module {
  NCA () {
  }

  torch::Tensor forward(torch::Tensor input) {
    return torch::zeros({16, 16});
  }

};

struct WorldDim {
  int channels;
  int height;
  int width;
};

int main(int argc, char* argv[]) {
  WorldDim dim{4, 8, 8};
  auto world = torch::zeros({dim.channels, dim.height, dim.width});
  float k[3][3] = {{0., 0.5, 0.}, {0.5, 1., 0.5}, {0., 0.5, 0.}};
  auto kernel = torch::tensor(k);
  std::cout << kernel << std::endl;

  torch::nn::Conv3d conv(torch::nn::Conv3dOptions(3, 3, 1).stride(1).bias(false));

  world.index_put_({2, Slice(), 5}, 1.0);
  world.index_put_({2, 5, Slice()}, 1.0);
  std::cout << world << std::endl;
  std::cout << "Done" << std::endl;
}
