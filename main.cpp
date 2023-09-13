#include <Eigen/Dense>
#include <png++/image.hpp>
#include <png++/require_color_space.hpp>

#include "header/eigen_image.h"
#include "utils.h"

//using namespace duho;

int main(int argc, char **argv)
{
    if (argc != 2 && argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image.png> [feature_size]" << std::endl;
        return 1;
    }

    // Read image
    std::string filename = argv[1];
    png::image<png::rgb_pixel > image(filename,  png::require_color_space<png::rgb_pixel>());

    // Convert RGB image to L*a*b* color space
    rgb_image_to_lab(image);

    // Superpixel generation algorithm
    double feature_size = argc == 3 ? std::stod(argv[2]) : image.get_width() * image.get_height() / 900.;
    double K = image.get_width() * image.get_height() / feature_size;
    double alpha = 2;
    uint8_t beta = 10;

    Eigen::MatrixXd matrix = duho::image_to_matrix<png::rgb_pixel>(image);

    std::cout << "Still working" << std::endl;

    return 0;
}
