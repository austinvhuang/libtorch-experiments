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

void log(std::string text, Tensor value) {
  std::cout << text << std::endl << value << std::endl;
}

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
  log("sobel_x1", sobel_x1);
  log("sobel_y1", sobel_y1);
  return std::make_pair(sobel_x1, sobel_y1); // do not use - values get corrupted
}

/* Model */

Tensor perceive(const Tensor& state_grid) {
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

  // auto [sobel_x, sobel_y] = create_sobel(4);
  const int duplicate_dim = 1;
  float sx[1][1][3][3] = {{{{-1., 0., 1.}, {-2., 0., 2.}, {-1., 0., 1.}}}};
  const auto sobel_x = torch::from_blob(sx, {1, 1, 3, 3});
  const auto sobel_y = torch::transpose(sobel_x, 3, -1);

  auto state_grid_flat = torch::transpose(state_grid, 0, 1);

//  std::cout << "sobel_x" << std::endl << sobel_x << std::endl;
//  std::cout << "sobel_y" << std::endl << sobel_x << std::endl;
//  std::cout << "state_grid_flat" << std::endl << state_grid_flat << std::endl;

  const auto grad_x =
      torch::transpose(torch::conv2d(state_grid_flat, sobel_x, {}, 1, 1), 0, 1);
  const auto grad_y =
      torch::transpose(torch::conv2d(state_grid_flat, sobel_y, {}, 1, 1), 0, 1);

  std::cerr << " sobel_x " << sobel_x.sizes() << std::endl;
  std::cerr << " sobel_y " << sobel_y.sizes() << std::endl;
  std::cerr << " grad_x " << grad_x.sizes() << std::endl;
  std::cerr << " grad_y " << grad_y.sizes() << std::endl;
  std::cerr << " state_ " << state_grid.sizes() << std::endl;
  std::vector<Tensor> perception_vec = {state_grid, grad_x, grad_y};
  auto perception_grid = torch::cat(perception_vec, 1);
  return perception_grid;
}

Tensor update(const Tensor& perception_vector, const Tensor& fc1,
              const Tensor& fc2) {
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
  std::cerr << "perception_vector " << perception_vector.sizes() << std::endl;
  std::cerr << "fc1 " << fc1.sizes() << std::endl;
  std::cerr << "fc2 " << fc2.sizes() << std::endl;

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
  auto update_threshold = 0.5;
  assert(state_grid.sizes() == ds_grid.sizes());
  auto rand_mask = torch::_cast_Float(rand_like(ds_grid).gt(update_threshold));
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
  std::cerr << "alpha " << alpha.sizes() << std::endl;
  auto alive =
      torch::max_pool2d(alpha, {3, 3}, {1}, {1}).gt(0.1); // stride=1, padding=1
  std::cerr << "alive pre " << alive.sizes() << std::endl;
  alive = torch::reshape(alive, {1, 1, alpha.sizes()[1], alpha.sizes()[2]});
  std::cerr << "alive post " << alive.sizes() << std::endl;
  alive = repeat_n(alive, state_grid.sizes()[1], 1);
  std::cerr << "alive rep " << alive.sizes() << std::endl;
  return state_grid * torch::_cast_Float(alive);
  return state_grid;
}

struct NCA : torch::nn::Module {
  NCA() {}
  Tensor forward(const Tensor& state_grid) {
    state_grid.to(torch::kCUDA);
    auto perception_grid = perceive(state_grid);

    // move channels to the last dimension for the dense layer computation
    auto perception_vec =
        torch::transpose(torch::transpose(perception_grid, 1, 2), 2, 3);
    std::cerr << "perception_vec " << perception_vec.sizes() << std::endl;

    auto ds = update(perception_vec, fc1, fc2);
    ds = torch::transpose(torch::transpose(ds, 2, 3), 1, 2);
    std::cerr << "state_grid " << state_grid.sizes() << std::endl;
    auto updated = stochastic_update(state_grid, ds);
    std::cerr << "updated    " << updated.sizes() << std::endl;
    // auto masked = alive_masking(updated);
    // return masked;
    return updated ;
  }
  Tensor fc1;
  Tensor fc2;
};

void train(NCA nca, Tensor target, Tensor init) {
  // TODO
  const int n_iter = 10000;
  const int t_max = 10;

  init.to(torch::kCUDA);
  target.to(torch::kCUDA);

//	torch::optim::SGD optimizer(
//		nca.parameters(),
//		torch::optim::SGDOptions(1e-5).momentum(0.5));
//
  for (int i = 0; i < n_iter; ++i) {
    Tensor state = init;
    state.to(torch::kCUDA);
    for (int t = 0; t < t_max; ++t) {
      state = nca.forward(state);
      std::cerr << state << std::endl;
    }
    auto loss = mse_loss(target, state);
    auto g = torch::autograd::grad({ loss }, { nca.fc1, nca.fc2 });
    if (i % 100 == 0) {
      std::cout << "================================================================================" << std::endl;
      std::cout << "iter: " << i << std::endl;
      std::cout << "loss: " << loss.to(torch::kFloat) << std::endl;
      std::cout << "state: " << state << std::endl;
      std::cout << "fc1 gradient norm: " << torch::norm(g[0]).to(torch::kFloat) << std::endl;
      std::cout << "fc2 gradient norm: " << torch::norm(g[1]).to(torch::kFloat) << std::endl;
    }

    float lr = 1e-36;
    nca.fc1 = nca.fc1 - lr * g[0];
    nca.fc2 = nca.fc2 - lr * g[1];
    std::cerr << "fc1: " << std::endl << nca.fc1 << std::endl;
    std::cerr << "fc2: " << std::endl << nca.fc2 << std::endl;
  }
}

int main(int argc, char *argv[]) {

  std::cout << "getNumGPUs: "  <<  torch::cuda::device_count() << std::endl;

  // make target pattern
  auto target = init_world(WorldDim{4, 6, 6});
  target.index_put_({0, 0, Slice(), 3}, 1.0); // 0 channel = red
  target.index_put_({0, 0, 3, Slice()}, 1.0);
  target.index_put_({0, 1, Slice(), 3}, 1.0); // 1 channel = green
  target.index_put_({0, 1, 3, Slice()}, 1.0);
  target.index_put_({0, 2, Slice(), 3}, 1.0); // 2 channel = blue
  target.index_put_({0, 2, 3, Slice()}, 1.0);
  target.index_put_({0, 3, Slice(), 3}, 1.0); // 3 channel = alpha
  target.index_put_({0, 3, 3, Slice()}, 1.0);
  std::cout << "target" << std::endl << target;

  auto init = init_world(WorldDim{4, 6, 6});
  init.index_put_({0, 0, 3, 3}, 1.0); // red channel
  init.index_put_({0, 1, 3, 3}, 1.0); // red channel
  init.index_put_({0, 2, 3, 3}, 1.0); // red channel
  init.index_put_({0, 3, 3, 3}, 1.0); // alpha
  std::cout << "init" << std::endl << init;

  std::cerr << "Target" << std::endl << target << std::endl;

  const int fc1_input_dim = (4 + 2 * 4); // add 2 * 4 for horizontal and vertical sobel filters
  auto nca = NCA();

  int hdim = 32;

  nca.fc1 = torch::zeros({hdim, fc1_input_dim}, torch::requires_grad()) + 1e-3; // avoid exactly 0 due to relu
  nca.fc1 = torch::nn::init::kaiming_uniform_(nca.fc1);
  nca.fc2 = torch::zeros({4, hdim}, torch::requires_grad());

  nca.fc1.to(torch::kCUDA);
  nca.fc2.to(torch::kCUDA);

  train(nca, target, init);

  std::cout << "Done" << std::endl;
}
