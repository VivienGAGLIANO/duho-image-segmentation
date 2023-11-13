#include <bit>
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

    // read image
    std::string filename = argv[1];
//    png::image<png::rgb_pixel> image(filename,  png::require_color_space<png::rgb_pixel>());
//
//    // prepare data
//    duho::augmented_matrix image_matrix = duho::prepare_data(image);
//
//    // parameters selection
//    double feature_size = argc == 3 ? std::stod(argv[2]) : image.get_width() * image.get_height() / 900.;
//    double K = image.get_width() * image.get_height() / feature_size;
//    K = pow(int(std::sqrt(K)), 2);
//    double alpha = 2;
//    uint8_t beta = 10;
//
//    // superpixel generation
//    duho::superpixel_generation superpixel_generation(image_matrix, feature_size, K);
//    std::vector<duho::superpixel> superpixels = superpixel_generation.generate_superpixels();
//    Eigen::MatrixXd clusters_image = superpixel_generation.clusters_to_image();
//    duho::write_image(clusters_image, {image.get_width(), image.get_height()}, filename, "output/", "_clusters");
//
//    // unseeded region-growing segmentation
//    duho::region_growing_segmentation region_growing_segmentation(superpixels, image_matrix);
//    std::vector<duho::region_growing_segmentation::region> regions = region_growing_segmentation.segment();
//    Eigen::MatrixXd regions_image = region_growing_segmentation.regions_to_image();
//    duho::write_image(regions_image, {image.get_width(), image.get_height()}, filename, "output/", "_regions");

    duho::test_parameters(1, 1024, filename, 1);

    return 0;
}
