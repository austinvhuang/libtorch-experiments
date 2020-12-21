#include <array>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <torch/torch.h>

using namespace torch::indexing;
using torch::Tensor;

#define USE_GPU
#ifdef USE_GPU
const auto options = torch::TensorOptions().device(torch::kCUDA, 0);
#else
const auto options = torch::TensorOptions();
#endif

const bool SAVE_OUTPUT = false;

struct WorldDim {
  int channels;
  int height;
  int width;
};

void logdat(std::string text, auto value) {
  std::cout << text << std::endl << value << std::endl;
}

void write_ppm(const Tensor img, const std::string &filename) {

  if (SAVE_OUTPUT) {

    std::ofstream ofile(filename);
    if (ofile.is_open()) {

      assert(img.sizes()[0] >= 4);
      const int img_width = img.sizes()[2];
      const int img_height = img.sizes()[1];

      ofile << "P3\n" << img_width << ' ' << img_height << "\n255\n";

      for (int i = 0; i < img_height; ++i) {
        for (int j = 0; j < img_width; ++j) {
          auto alpha = 1.0; // img[3][i][j];
          float r = (torch::relu(img[0][i][j] * alpha)).item<float>();
          float g = (torch::relu(img[1][i][j] * alpha)).item<float>();
          float b = (torch::relu(img[2][i][j] * alpha)).item<float>();
          int ir = static_cast<int>(255.999 * r);
          int ig = static_cast<int>(255.999 * g);
          int ib = static_cast<int>(255.999 * b);

          ofile << ir << ' ' << ig << ' ' << ib << '\n';
        }
      }
    }
  }
}

Tensor repeat_n(const Tensor t, const int n, const int dim) {
  // repeat a tensor n times along dimension dim
  // used to replicate sobel filter across all channels
  std::vector<Tensor> tvec(n, t);
  return torch::cat(tvec, dim);
}

Tensor init_world(const int batchsize, const WorldDim dim) {
  return torch::zeros({batchsize, dim.channels, dim.height, dim.width},
                      options);
}

/* Model */

struct NCA : torch::nn::Module {
  NCA(int fc1_input_dim, int hdim, int channels, int batchsize)
      : fc1(torch::nn::Conv2d(fc1_input_dim, hdim, /*kernel size*/ 1)),
        fc2(torch::nn::Conv2d(hdim, channels, /*kernel size */ 1)) {

    fc1->weight =
        torch::zeros({hdim, fc1_input_dim, 1, 1}, options.requires_grad(true));
    fc1->bias = torch::zeros({hdim}, options.requires_grad(true));
    fc1->weight = torch::nn::init::kaiming_uniform_(fc1->weight);
    fc2->weight =
        torch::full({channels, hdim, 1, 1}, 1e-5, options.requires_grad(true));
    fc2->bias = torch::full({channels}, 1e-5, options.requires_grad(true));

    register_parameter("fc1_weight", fc1->weight);
    register_parameter("fc1_b", fc1->bias);
    register_parameter("fc2_weight", fc2->weight);
    register_parameter("fc2_b", fc2->bias);

    register_module("fc1", fc1); // must come after reassignment?
    register_module("fc2", fc2);

    float sx[1][1][3][3] = {{{{-1., 0., 1.}, {-2., 0., 2.}, {-1., 0., 1.}}}};
    const auto sobel_x0 = torch::from_blob(sx, {1, 1, 3, 3}) /
                          8.0; // divide by 8.0 to conserve intensity
    sobel_x =
        sobel_x0.to(options); // cannot initialize on the gpu using from_blob
    sobel_y = torch::transpose(sobel_x, 3, -1);

    sobel_x = repeat_n(sobel_x, channels, 0);
    sobel_y = repeat_n(sobel_y, channels, 0);
    sobel_x = repeat_n(sobel_x, channels, 1);
    sobel_y = repeat_n(sobel_y, channels, 1);
  }

  Tensor perceive(const Tensor &state_grid) {
    auto grad_x = 
        torch::conv2d(state_grid, sobel_x, {}, 1, 1);
    auto grad_y = torch::transpose(
        torch::conv2d(state_grid, sobel_y, {}, 1, 1), 2, 3);
    std::vector<Tensor> perception_vec = {state_grid, grad_x, grad_y};
    auto perception_grid = torch::cat(perception_vec, 1);
    return perception_grid;
  }

  Tensor update(const Tensor &perception_vector) {
    auto x = fc1->forward(perception_vector);
    x = torch::relu(x);
    auto ds = fc2->forward(x);
    return ds;
  }

  Tensor stochastic_update(const Tensor &state_grid, const Tensor &ds_grid) {
    const float update_threshold = 0.5;
    assert(state_grid.sizes() == ds_grid.sizes());
    auto rand_mask =
        torch::_cast_Float(rand_like(ds_grid).gt(update_threshold));
    auto ds_masked = ds_grid * rand_mask;
    return state_grid + ds_masked;
  }

  Tensor alive_masking(const Tensor &state_grid) {
    const float threshold = 0.1;
    auto alpha =
        state_grid.index({Slice(), 3, Slice(), Slice()}); // 3 = alpha channel
    auto alive = torch::max_pool2d(alpha, {3, 3}, {1}, {1})
                     .gt(threshold); // stride=1, padding=1
    alive = torch::reshape(alive, {state_grid.sizes()[0], 1, alpha.sizes()[1], alpha.sizes()[2]});
    alive = repeat_n(alive, state_grid.sizes()[1],
                     1); // repeat alive mask all channels
    return state_grid * torch::_cast_Float(alive);
  }

  Tensor forward(Tensor &state_grid) {
    // std::cout << "is cuda? " << state_grid.is_cuda() << std::endl;
    auto perception_grid = perceive(state_grid);
    auto ds = update(perception_grid);
    auto updated = stochastic_update(state_grid, ds);
    auto masked = alive_masking(updated);
    return masked;
  }

  Tensor sobel_x;
  Tensor sobel_y;
  torch::nn::Conv2d fc1;
  torch::nn::Conv2d fc2;
};

std::string make_outdir() {

  auto tm = time(NULL);
  auto ltm = localtime(&tm);
  std::stringstream ss;
  ss << (ltm->tm_year + 1900) << std::setfill('0') << std::setw(2)
     << ltm->tm_mon << std::setw(2) << ltm->tm_mday << "_" << std::setw(2)
     << ltm->tm_hour << std::setw(2) << ltm->tm_min;
  std::string outdir = "images_" + ss.str();

  if (SAVE_OUTPUT) {
    std::filesystem::create_directory(outdir);
  }

  return outdir;
}

void train(NCA &nca, const Tensor &target, const Tensor &init, const float &lr,
           const int tmax, const std::string outdir) {
  const int n_iter = 10000000;

  assert(target.sizes()[1] >= 4);
  assert(init.sizes()[1] >= 4);
  // torch::optim::Adam optimizer(nca.parameters(), torch::optim::AdamOptions(lr));
  // torch::optim::SGD optimizer(nca.parameters(), torch::optim::SGDOptions(lr).momentum(0.5));
  // logdat("lr check ", lr);
  // logdat("object check ", nca.parameters());
  torch::optim::SGD optimizer(nca.parameters(), torch::optim::SGDOptions(lr).momentum(0.0));
  // logdat("optim ", optimizer);
    

  auto target_img = target.index({0, Slice(), Slice(), Slice()});
  write_ppm(target_img, outdir + "/target.ppm");

  nca.train();

  for (int i = 0; i < n_iter; ++i) {
    optimizer.zero_grad();
    Tensor state = init;
    state = state.to(options);
    for (int t = 0; t < tmax; ++t) {
      state = nca.forward(state);
    }
    // auto loss = mse_loss(target.slice(1, 0, 4), state.slice(1, 0, 4));
    auto loss = mse_loss(state.slice(1, 0, 4), target.slice(1, 0, 4));
    // logdat("prior ", torch::norm(nca.fc2->weight));
    auto prior = nca.fc2->weight;
    loss.backward();
    optimizer.step();
    // logdat("after ", torch::norm(nca.fc2->weight));
//    std::cout << "fc2 gradient norm: "
//                << torch::norm(nca.fc2->weight.grad()).to(torch::kFloat)
//                << std::endl;
//    logdat("diff", torch::norm(nca.fc2->weight - prior));
    // auto g = torch::autograd::grad({loss}, {nca.fc1, nca.fc2});
    if (i % 10 == 0) {
      std::cout << "iter: " << i << " | loss: " << std::setprecision(5)
                << loss.item<float>() << "\n";
    }
    if (i % 500 == 0) {
      std::cout << "========================================================="
                   "======================="
                << std::endl;
      std::cout << "iter: " << i << std::endl;
      std::cout << "loss: " << loss.to(torch::kFloat) << std::endl;
      std::cout << "state (visible): " << state.slice(1, 0, 4).slice(0, 0, 1) << std::endl;
      std::cout << "state dtype: " << state.is_cuda() << std::endl;
      std::cout << "fc1 gradient norm: "
                << torch::norm(nca.fc1->weight.grad()).to(torch::kFloat)
                << std::endl;
      std::cout << "fc2 gradient norm: "
                << torch::norm(nca.fc2->weight.grad()).to(torch::kFloat)
                << std::endl;

      std::cout << "Writing images\n";
      Tensor img = init;
      img = img.to(options);
      for (int t = 0; t < tmax; ++t) {
        auto state_img =
            img.index({0, Slice(), Slice(), Slice()}); // 3 = alpha channel
        std::stringstream ss;
        ss << outdir << "/state_" << std::setfill('0') << std::setw(8) << i
           << "_" << std::setw(3) << t << ".ppm";
        write_ppm(state_img, ss.str());
        img = nca.forward(img);
      }
      std::cout << "Wrote images\n";
    }
  }
}

int main(int argc, char *argv[]) {

  std::cout << "getNumGPUs: " << torch::cuda::device_count() << std::endl;

  const int batchsize = 1;
  auto world_dim = WorldDim({16, 9, 9});
  const int hdim = 128;
  const float lr = 1.0e-4;
  const int tmax = 70;
  auto outdir = make_outdir();

  // make target pattern
  auto target = init_world(batchsize, world_dim);
  target.index_put_({0, 0, Slice(), 4}, 1.0); // 0 channel = red
  target.index_put_({0, 0, 4, Slice()}, 1.0);
  target.index_put_({0, 1, Slice(), 4}, 1.0); // 1 channel = green
  target.index_put_({0, 1, 4, Slice()}, 1.0);
  target.index_put_({0, 2, Slice(), 4}, 1.0); // 2 channel = blue
  target.index_put_({0, 2, 4, Slice()}, 1.0);
  target.index_put_({0, 3, Slice(), 4}, 1.0); // 3 channel = alpha
  target.index_put_({0, 3, 4, Slice()}, 1.0);
  logdat("target", target);

  auto init = init_world(batchsize, world_dim);
  init.index_put_({0, Slice(), 4, 4}, 1.0); // alpha
  logdat("init", init);

  const int fc1_input_dim =
      (world_dim.channels +
       2 * world_dim.channels); // add 2 * 4 for horizontal and vertical sobel
                                // filters
  auto nca = NCA(fc1_input_dim, hdim, world_dim.channels, batchsize);
  torch::Device device(torch::kCUDA);
  nca.to(device);

  std::cout << "hdim: " << hdim << std::endl;
  std::cout << "lr: " << lr << std::endl;
  std::cout << "tmax: " << tmax << std::endl;
  std::cout << "world channels: " << world_dim.channels << std::endl;
  std::cout << "outdir: " << outdir << std::endl;

  train(nca, target, init, lr, tmax, outdir);

  std::cout << "Done" << std::endl;
}
