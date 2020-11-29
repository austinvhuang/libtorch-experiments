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
  std::vector<Tensor> tvec(n, t);
  return torch::cat(tvec, dim);
}

Tensor init_world(const WorldDim dim) {
  return torch::zeros({1, dim.channels, dim.height, dim.width});
}

std::pair<Tensor, Tensor> create_sobel(int n_channels) {
  const int duplicate_dim = 1;
  float sx[1][1][3][3] = {{{{-1., 0., 1.}, {-2., 0., 2.}, {-1., 0., 1.}}}};
  const auto sobel_x1 = torch::from_blob(sx, {1, 1, 3, 3});
  const auto sobel_y1 = torch::transpose(sobel_x1, 3, -1);
  return {sobel_x1, sobel_y1};
}

/* Model */

Tensor perceive(const Tensor state_grid) {
  // This step defines what each cell perceives of the environment surrounding
  // it.
  //
  // We implement this via a 3x3 convolution with a fixed kernel.
  // we are using classical Sobel filters to estimate the partial derivatives
  // forming a 2D gradient vector in each direction, for each state channel.
  //
  // We concatenate those gradients with the cells own states, forming a
  // percepted vector, for each cell.

  assert(state_grid.sizes()[0] == 1);

  auto [sobel_x, sobel_y] = create_sobel(4);

  auto state_grid_flat = torch::transpose(state_grid, 0, 1);

  const auto grad_x =
      torch::transpose(torch::conv2d(state_grid_flat, sobel_x, {}, 1, 1), 0, 1);
  const auto grad_y =
      torch::transpose(torch::conv2d(state_grid_flat, sobel_y, {}, 1, 1), 0, 1);

  std::cout << " sobel_x " << sobel_x.sizes() << std::endl;
  std::cout << " sobel_y " << sobel_y.sizes() << std::endl;
  std::cout << " grad_x " << grad_x.sizes() << std::endl;
  std::cout << " grad_y " << grad_y.sizes() << std::endl;
  std::cout << " state_ " << state_grid.sizes() << std::endl;
  std::vector<Tensor> perception_vec = {state_grid, grad_x, grad_y};
  auto perception_grid = torch::cat(perception_vec, 1);
  std::cout << " perception_grid " << perception_grid.sizes() << std::endl;
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
  std::cout << "perception_vector " << perception_vector.sizes() << std::endl;
  std::cout << "fc1 " << fc1.sizes() << std::endl;
  std::cout << "fc2 " << fc2.sizes() << std::endl;

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
  //
  assert(state_grid.sizes() == ds_grid.sizes());
  auto rand_mask = torch::_cast_Float(rand_like(ds_grid).gt(0.5));
  auto ds_masked = ds_grid * rand_mask;
  return state_grid + ds_masked;
}

Tensor alive_masking(const Tensor state_grid) {
  // we don’t want empty cells to participate in computations or carry any
  // hidden state.
  //
  // We enforce this by explicitly setting all channels of empty cells to zeros.
  // A cell is considered empty if there is no “mature” (alpha>0.1) cell in its
  // 3x3 neightborhood.
  auto alpha =
      state_grid.index({Slice(), 3, Slice(), Slice()}); // 3 = alpha channel
  std::cout << "alpha " << alpha.sizes() << std::endl;
  auto alive =
      torch::max_pool2d(alpha, {3, 3}, {1}, {1}).gt(0.1); // stride=1, padding=1
  std::cout << "alive pre " << alive.sizes() << std::endl;
  alive = torch::reshape(alive, {1, 1, alpha.sizes()[1], alpha.sizes()[2]});
  std::cout << "alive post " << alive.sizes() << std::endl;
  alive = repeat_n(alive, state_grid.sizes()[1], 1);
  std::cout << "alive rep " << alive.sizes() << std::endl;
  return state_grid * torch::_cast_Float(alive);
  return state_grid;
}

struct NCA : torch::nn::Module {
  NCA() {}
  Tensor forward(Tensor state_grid) {
    auto perception_grid = perceive(state_grid);

    // move channels to the last dimension for the dense layer computation
    auto perception_vec =
        torch::transpose(torch::transpose(perception_grid, 1, 2), 2, 3);
    std::cout << "perception_vec " << perception_vec.sizes() << std::endl;

    auto ds = update(perception_vec, fc1, fc2);
    ds = torch::transpose(torch::transpose(ds, 2, 3), 1, 2);
    std::cout << "state_grid " << state_grid.sizes() << std::endl;
    std::cout << "ds         " << ds.sizes() << std::endl;
    auto updated = stochastic_update(state_grid, ds);
    std::cout << "updated    " << updated.sizes() << std::endl;
    auto masked = alive_masking(updated);
    return masked;
  }
  Tensor fc1;
  Tensor fc2;
};

void train(NCA nca, Tensor world) {
  // TODO
  const int t_max = 3;
  for (int t = 0; t < t_max; ++t) {
    world = nca.forward(world);
    std::cout << world << std::endl;
  }
}

int main(int argc, char *argv[]) {
  auto world = init_world(WorldDim{4, 8, 8});
  world.index_put_({0, 0, Slice(), 5}, 1.0); // red
  world.index_put_({0, 0, 5, Slice()}, 1.0);
  world.index_put_({0, 3, Slice(), 5}, 1.0); // alpha
  world.index_put_({0, 3, 5, Slice()}, 1.0);
  std::cout << "World" << std::endl << world << std::endl;

  const int fc1_input_dim = (4 + 2 * 4);
  auto nca = NCA();
  nca.fc1 = torch::zeros({128, fc1_input_dim});
  nca.fc2 = torch::zeros({4, 128});

  train(nca, world);

  std::cout << "Done" << std::endl;
}
