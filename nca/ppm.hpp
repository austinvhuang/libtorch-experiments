#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <torch/torch.h>

using torch::Tensor;

struct RGB {
  int r;
  int g;
  int b;
};

struct PPM {
  int columns;
  int rows;
  int max;
  std::vector<RGB> pixels;
};

void write_ppm(const Tensor img, const std::string &filename) {
  const auto cpu_options = torch::TensorOptions().device(torch::kCPU);
    std::ofstream ofile(filename);
    if (ofile.is_open()) {
      assert(img.sizes()[0] >= 4);
      const int img_width = img.sizes()[2];
      const int img_height = img.sizes()[1];
      ofile << "P3\n" << img_width << ' ' << img_height << "\n255\n";
      for (int i = 0; i < img_height; ++i) {
        for (int j = 0; j < img_width; ++j) {
          auto rimg = (img[0][i][j]).to(cpu_options);
          auto gimg = (img[1][i][j]).to(cpu_options);
          auto bimg = (img[2][i][j]).to(cpu_options);
          auto alpha = img[3][i][j].to(cpu_options);
          float r = torch::min(torch::full(1, 1.0, cpu_options),
                               (torch::relu(rimg * alpha)))
                        .item<float>();
          float g = torch::min(torch::full(1, 1.0, cpu_options),
                               (torch::relu(gimg * alpha)))
                        .item<float>();
          float b = torch::min(torch::full(1, 1.0, cpu_options),
                               (torch::relu(bimg * alpha)))
                        .item<float>();
          int ir = static_cast<int>(255.999 * r);
          int ig = static_cast<int>(255.999 * g);
          int ib = static_cast<int>(255.999 * b);
          ofile << ir << ' ' << ig << ' ' << ib << '\n';
        }
      }
    }
}

std::shared_ptr<PPM> read_ppm(const std::string& infile) {
  // warning - no validation
  std::shared_ptr<PPM> ppm = std::make_shared<PPM>();

  std::ifstream img(infile);
  if (img.is_open()) {
    int token_index = 0;

    std::string line;
    while (token_index <= 3 && std::getline(img, line)) {
      std::stringstream line_stream(line);
      std::cout << "\nline is \n" << line << std::endl;
      std::string token;
      while (line_stream >> token) {

        if (token == "#") {
          break;
        }
        if (token_index == 0) {
          if (token != "P3") {
            std::cout << "Not a P3 file\n";
            exit(1);
          }
        }
        if (token_index == 1) {
          ppm->columns = stoi(token);
        }
        if (token_index == 2) {
          ppm->rows = stoi(token);
        }
        if (token_index == 3) {
          ppm->max = stoi(token);
        }
        ++token_index;
      }
    }

    ppm->pixels = std::vector<RGB>(ppm->columns * ppm->rows);
    token_index = 0;
    while (std::getline(img, line)) {
      std::string token;
      std::stringstream line_stream(line);
      while (line_stream >> token) {
        int pixel_index = token_index / 3;
        if (token_index % 3 == 0) {
          ppm->pixels[pixel_index].r = stoi(token);
        }
        if (token_index % 3 == 1) {
          ppm->pixels[pixel_index].g = stoi(token);
        }
        if (token_index % 3 == 2) {
          ppm->pixels[pixel_index].b = stoi(token);
        }
        ++token_index;
      }
    }

    img.close();
  }

  return ppm;
}
