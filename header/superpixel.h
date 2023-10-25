#ifndef DUHO_SUPERPIXEL_H
#define DUHO_SUPERPIXEL_H

#include <vector>

#include "Eigen/Dense"
#include "eigen_image.h"

namespace duho
{

class superpixel
{
public:
    // constructors and destructors
    superpixel(const std::vector<Eigen::Vector2d> &pixels, augmented_matrix &image);
    superpixel operator=(const superpixel &sp);

//    ~superpixel()=default;

    void add_pixel(const Eigen::Vector2d &pixel);


    static bool connected(const superpixel &sp1, const superpixel &sp2);
    static double dist_1(const Eigen::Vector2d &u, const Eigen::Vector2d &v);

//private:
    std::vector<Eigen::Vector2d> m_pixels;
    Eigen::Vector3d m_mean; // this should always be updated with pixels currently present in the superpixel

    augmented_matrix &m_image;
};

} // duho

#endif //DUHO_SUPERPIXEL_H
