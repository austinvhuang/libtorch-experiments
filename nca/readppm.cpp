#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "ppm.hpp"


int main() {
  auto ppm = read_ppm("input.ppm");
  std::cout << "columns: " << ppm->columns << "\nrows: " << ppm->rows
            << "\nmax: " << ppm->max << "\npixels:\n";

  for (auto pixel : ppm->pixels) {
    std ::cout << "r: " << pixel.r << "\ng: " << pixel.g << "\nb: " << pixel.b
               << std::endl;
  }
}
