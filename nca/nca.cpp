#include <array>
#include <iostream>
#include <torch/torch.h>

using namespace torch::indexing;
using torch::Tensor;

struct WorldDim {
  int channels;
  int height;
  int width;
};

/* helper operations */

Tensor alpha2state(const Tensor alpha) {
  // cells having \alpha > 0.1α>0.1 and their neighbors are considered
  // “living”. Other cells are “dead” or empty and have their state vector
  // values explicitly set to 0.0 at each time step.
  return torch::sign(alpha - 0.1) / 2.0 + 0.5;
}

Tensor repeat_n(const Tensor t, const int n, const int dim) {
  // repeat a tensor n times along dimension dim
  // used to replicate sobel filter across all channels
  std::vector<Tensor> tvec;
  for (auto i = 0; i < n; i++) {
    tvec.push_back(t);
  }
  return torch::cat(tvec, dim);
}

Tensor init_world(const WorldDim dim) {
  return torch::zeros({1, dim.channels, dim.height, dim.width});
}

std::pair<Tensor, Tensor> create_sobel(int n_channels) {
  const int duplicate_dim = 1;
  float sx[1][1][3][3] = {{{{-1., 0., 1.}, {-2., 0., 2.}, {-1., 0., 1.}}}};
  auto sobel_x1 = torch::from_blob(sx, {1, 1, 3, 3});
  auto sobel_y1 = torch::transpose(sobel_x1, 0, -1);
  return {repeat_n(sobel_x1, n_channels, duplicate_dim),
          repeat_n(sobel_y1, n_channels, duplicate_dim)};
}

/* Model */

Tensor perceive(Tensor state_grid) {
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
  std::vector<Tensor> perception_vec = {state_grid, grad_x, grad_y};
  // auto perception_grid = torch::cat(perception_vec, 2);
  // TODO fix padding dimensions
  auto perception_grid = state_grid;
  return perception_grid;
}

Tensor update(const Tensor perception_vector, const Tensor fc1,
              const Tensor fc2) {
  // Each cell now applies a series of operations to the perception vector which
  // we call the cell’s “update rule”.
  //
  // Every cell runs the same update rule. The update rule outputs an
  // incremental update to the cell’s state (ds), which applied to the cell
  // before the next time step.
  //
  // The update rule is designed to exhibit “do-nothing” initial behaviour -
  // implemented by initializing the weights of the final convolutional layer in
  // the update rule with zero.
  auto x = torch::linear(perception_vector, fc1);
  x = torch::relu(x);
  auto ds = torch::linear(x, fc2);
  return ds;
}

Tensor stochastic_update(const Tensor state_grid, const Tensor ds_grid) {
  // we apply a random per-cell mask to update vectors, setting all update
  // values to 0with some predefined probability (we use 0.5 during training).
  // This operation can be also seen as an application of per-cell dropout to
  // update vectors.
  auto rand_mask = torch::_cast_Float(rand_like(state_grid).gt(0.5));
  return state_grid + ds_grid;
}

Tensor alive_masking(const Tensor state_grid) {
  // we don’t want empty cells to participate in computations or carry any
  // hidden state.
  //
  // We enforce this by explicitly setting all channels of empty cells to zeros.
  // A cell is considered empty if there is no “mature” (alpha>0.1) cell in its
  // 3x3 neightborhood.
  auto alive =
      torch::max_pool2d(state_grid.index({Slice(), Slice(), 3}), {3, 3})
          .gt(0.1);
  return state_grid * torch::_cast_Float(alive);
}

struct NCA : torch::nn::Module {
  NCA() {}
  Tensor forward(Tensor state_grid) {
    auto perception_vector = perceive(state_grid);
    auto ds = update(perception_vector, fc1, fc2);
    auto updated = stochastic_update(state_grid, ds);
    auto masked = alive_masking(updated);
    return masked;
  }
  Tensor fc1 = torch::zeros({16, 16});
  Tensor fc2 = torch::zeros({16, 16});
};

int main(int argc, char *argv[]) {
  auto world = init_world(WorldDim{4, 8, 8});

  auto [sobel_x, sobel_y] = create_sobel(4);

  std::cout << "sobel_x: " << std::endl << sobel_x << std::endl;
  std::cout << "sobel_y: " << std::endl << sobel_y << std::endl;

  world.index_put_({0, 0, Slice(), 5}, 1.0);
  world.index_put_({0, 0, 5, Slice()}, 1.0);

  std::cout << world << std::endl;

  auto nca = NCA();

  auto out = perceive(world);

  std::cout << out << std::endl;

  std::cout << "Done" << std::endl;
}
