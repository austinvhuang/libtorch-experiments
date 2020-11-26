#include <array>
#include <iostream>
#include <torch/torch.h>

using namespace torch::indexing;

struct NCA : torch::nn::Module {
  NCA() {}

  torch::Tensor forward(torch::Tensor input) { return torch::zeros({16, 16}); }
};

struct WorldDim {
  int channels;
  int height;
  int width;
};

torch::Tensor alpha2state(const torch::Tensor alpha) {
  // cells having \alpha > 0.1α>0.1 and their neighbors are considered “living”.
  // Other cells are “dead” or empty and have their state vector values
  // explicitly set to 0.0 at each time step.
  return torch::sign(alpha - 0.1) / 2.0 + 0.5;
}

torch::Tensor repeat_n(const torch::Tensor t, const int n, const int dim) {
  // repeat a tensor n times along dimension dim
  // used to replicate sobel filter across all channels
  std::vector<torch::Tensor> tvec;
  for (auto i = 0; i < n; i++) {
    tvec.push_back(t);
  }
  return torch::cat(tvec, dim);
}

torch::Tensor init_world(const WorldDim dim) {
  return torch::zeros({1, dim.channels, dim.height, dim.width});
}

std::pair<torch::Tensor, torch::Tensor> create_sobel(int n_channels) {
  const int duplicate_dim = 1;
  float sx[1][1][3][3] = {{{{-1., 0., 1.}, {-2., 0., 2.}, {-1., 0., 1.}}}};
  auto sobel_x1 = torch::from_blob(sx, {1, 1, 3, 3});
  auto sobel_y1 = torch::transpose(sobel_x1, 0, -1);
  return {repeat_n(sobel_x1, n_channels, duplicate_dim),
          repeat_n(sobel_y1, n_channels, duplicate_dim)};
}

torch::Tensor perceive(torch::Tensor state_grid) {
  // This step defines what each cell perceives of the environment surrounding
  // it.
  //
  // We implement this via a 3x3 convolution with a fixed kernel.
  // we are using classical Sobel filters to estimate the partial derivatives
  // forming a 2D gradient vector in each direction, for each state channel.
  //
  // We concatenate those gradients with the cells own states, forming a
  // percepted vector, for each cell.
  auto [sobel_x, sobel_y] = create_sobel(4);
  auto grad_x = torch::conv2d(state_grid, sobel_x);
  auto grad_y = torch::conv2d(state_grid, sobel_y);
  std::vector<torch::Tensor> perception_vec = {state_grid, grad_x, grad_y};
  // auto perception_grid = torch::cat(perception_vec, 2);
  // TODO fix padding dimensions
  auto perception_grid = state_grid;
  return perception_grid;
}

int main(int argc, char *argv[]) {
  auto world = init_world(WorldDim{4, 8, 8});

  auto [sobel_x, sobel_y] = create_sobel(4);

  std::cout << "sobel_x: " << std::endl << sobel_x << std::endl;
  std::cout << "sobel_y: " << std::endl << sobel_y << std::endl;

  world.index_put_({0, 0, Slice(), 5}, 1.0);
  world.index_put_({0, 0, 5, Slice()}, 1.0);

  std::cout << world << std::endl;

  auto out = perceive(world);

  auto outx = torch::conv2d(world, sobel_x, {}, 1, 0, 1, 1);
  auto outy = torch::conv2d(world, sobel_x, {}, 1, 0, 1, 1);
  std::cout << outx << std::endl;
  std::cout << outy << std::endl;
  std::cout << out << std::endl;

  std::cout << "Done" << std::endl;
}
