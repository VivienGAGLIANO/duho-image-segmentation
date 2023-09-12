#ifndef DUHO_ALGORITHM_H
#define DUHO_ALGORITHM_H

#include <vector>

#include "Eigen/Core"
#include "superpixel.h"

namespace duho
{

    class superpixel_generation
    {
    public:
        superpixel_generation();
        ~superpixel_generation();

        static std::vector<superpixel> generate_superpixels(const Eigen::MatrixXd &image);
    };


    class region_growing_segmentation
    {
    public:
        region_growing_segmentation();
        ~region_growing_segmentation();

        static std::vector<superpixel> segment(const Eigen::MatrixXd &image, const std::vector<superpixel> &superpixels);
    };

} // duho

#endif //DUHO_ALGORITHM_H
