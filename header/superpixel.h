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


    static bool connected(const superpixel &sp1, const superpixel &sp2);
    static double dist_1(const Eigen::Vector2d &u, const Eigen::Vector2d &v);

//private:
    std::vector<Eigen::Vector2d> m_pixels;
};

} // duho

#endif //DUHO_SUPERPIXEL_H
