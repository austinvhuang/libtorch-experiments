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
  int depth ;
  int height;
  int width;
};

int main(int argc, char* argv[]) {
  WorldDim dim{4, 8, 8}; 
  auto world = torch::zeros({1, 1, dim.height, dim.width});

  auto options = torch::TensorOptions();

  float sx[1][1][3][3] = {{{{-1., 0., 1.}, {-2., 0., 2.}, {-1., 0., 1.}}}};
  auto sobel_x = torch::from_blob(sx, {1, 1, 3, 3});
  auto sobel_y = torch::transpose(sobel_x, 0, -1);

  std::cout << "sobel_x: " << std::endl << sobel_x << std::endl;
  std::cout << "sobel_y: " << std::endl << sobel_y << std::endl;
  world.index_put_({0, 0, Slice(), 5}, 1.0);
  world.index_put_({0, 0, 5, Slice()}, 1.0);

  std::cout << world << std::endl;

  auto out = torch::conv2d(world, sobel_x, {}, 1, 0, 1, 1);
  std::cout << out << std::endl;
  std::cout << "Done" << std::endl;
}
