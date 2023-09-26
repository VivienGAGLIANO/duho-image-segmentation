#include <Eigen/Dense>
#include <png++/image.hpp>
#include <png++/require_color_space.hpp>

#include "algorithm.h"
#include "eigen_image.h"
#include "utils.h"


int main(int argc, char **argv)
{
    if (argc != 2 && argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image.png> [feature_size]" << std::endl;
        return 1;
    }

    // Read image
    std::string filename = argv[1];
    png::image<png::rgb_pixel> image(filename,  png::require_color_space<png::rgb_pixel>());

    // Convert RGB image to L*a*b* color space
    rgb_image_to_lab(image);

    // Superpixel generation algorithm
    double feature_size = argc == 3 ? std::stod(argv[2]) : image.get_width() * image.get_height() / 900.;
    double K = image.get_width() * image.get_height() / feature_size;
    double alpha = 2;
    uint8_t beta = 10;

    Eigen::MatrixXd matrix = duho::image_to_matrix<png::rgb_pixel>(image);
    duho::superpixel_generation superpixel_generation(matrix, feature_size, K);
    superpixel_generation.generate_superpixels();
//    duho::matrix_to_image<png::rgb_pixel>(superpixel_generation.m_image*255., {image.get_width(), image.get_height()}).write(filename.replace(0, 10, "output/").substr(0, filename.size()-4) + "_normalized.png");
//    image.write("albedo_Lab.png");
    Eigen::MatrixXd clusters = superpixel_generation.clusters_to_image();
    duho::matrix_to_image<png::rgb_pixel>(clusters, {image.get_width(), image.get_height()}).write(filename.replace(0, 10, "output/").substr(0, filename.size()-4) + "_clusters.png");

    std::cout << "Still working" << std::endl;

    return 0;
}
