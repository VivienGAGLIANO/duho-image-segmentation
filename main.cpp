#include <bit>
#include <Eigen/Dense>
#include <png++/image.hpp>
#include <png++/require_color_space.hpp>

#include "algorithm.h"
#include "eigen_image.h"
#include "utils.h"


int main(int argc, char **argv)
{
    if (argc != 4 && argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <image.png> <background seed x> <background seed y> [feature_size]" << std::endl;
        return 1;
    }

    // read image
    std::string filename = argv[1];

    // TODO pre-process region growing pass to isolate background and remove it from further classification

    png::image<png::rgb_pixel> image(filename,  png::require_color_space<png::rgb_pixel>());

    // prepare data
    duho::augmented_matrix image_matrix = duho::prepare_data(image);

    // parameters selection
    double feature_size = argc == 5 ? std::stod(argv[4]) : image.get_width() * image.get_height() / 900.;
    double K = image.get_width() * image.get_height() / feature_size;
    K = pow(int(std::sqrt(K)), 2);
    double alpha = 2;
    uint8_t beta = 10;

    // detect background
    Eigen::Vector2d seed = {std::stod(argv[2]), std::stod(argv[3])};
    std::vector<Eigen::Vector2d> background = duho::detect_background(image_matrix, seed, .195);
    for (const Eigen::Vector2d &pixel : background)
        image[pixel.y()][pixel.x()] = png::rgb_pixel(255, 0, 0);
    duho::write_image(duho::image_to_matrix(image), {image.get_width(), image.get_height()}, filename, "_background");

//    // superpixel generation
//    duho::superpixel_generation superpixel_generation(image_matrix, feature_size, K);
//    std::vector<duho::superpixel> superpixels = superpixel_generation.generate_superpixels();
//    Eigen::MatrixXd clusters_image = superpixel_generation.clusters_to_image();
//    duho::write_image(clusters_image, {image.get_width(), image.get_height()}, filename, "_clusters");
//
//    // unseeded region-growing segmentation
//    duho::region_growing_segmentation region_growing_segmentation(superpixels, image_matrix);
//    std::vector<duho::region_growing_segmentation::region> regions = region_growing_segmentation.segment();
//    Eigen::MatrixXd regions_image = region_growing_segmentation.regions_to_image();
//    duho::write_image(regions_image, {image.get_width(), image.get_height()}, filename, "_regions");

//    duho::test_parameters(100, 150, filename, 1, true);

    return 0;
}
