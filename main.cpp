#include <Eigen/Dense>
#include <png++/image.hpp>
#include <png++/require_color_space.hpp>

#include "header/eigen_image.h"
#include "utils.h"

//using namespace duho;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image.png>" << std::endl;
        return 1;
    }

    // Read image
    std::string filename = argv[1];
    png::image<png::rgb_pixel > image(filename,  png::require_color_space<png::rgb_pixel>());

    // Convert RGB image to L*a*b* color space
    rgb_image_to_lab(image);
//    image.write("lab.png");


    Eigen::MatrixXd matrix = duho::image_to_matrix<png::rgb_pixel>(image);

    std::cout << "Still working" << std::endl;

    return 0;
}
