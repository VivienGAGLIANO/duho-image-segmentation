#ifndef DUHO_SUPERPIXEL_H
#define DUHO_SUPERPIXEL_H

#include <vector>
#include "Eigen/Dense"

namespace duho
{

class superpixel
{
public:
    // constructors and destructors
    superpixel();
    explicit superpixel(const std::vector<Eigen::Vector2d> &pixels);

    ~superpixel();

    void add_pixel(const Eigen::Vector2d &pixel);


    static bool connected(const superpixel &sp1, const superpixel &sp2);
    static double dist_1(const Eigen::Vector2d &u, const Eigen::Vector2d &v);

//private:
    std::vector<Eigen::Vector2d> m_pixels;
    Eigen::Vector3d m_mean; // this should always be updated with pixels currently present in the superpixel
    int q1, q2, q3;
    double iqr;
};

} // duho

#endif //DUHO_SUPERPIXEL_H
