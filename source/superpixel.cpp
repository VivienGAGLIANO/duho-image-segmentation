#include "superpixel.h"

namespace duho
{

superpixel::superpixel() : m_pixels() {}

superpixel::superpixel(const std::vector<Eigen::Vector2d> &pixels) : m_pixels(pixels) {}

superpixel::~superpixel()
{
    m_pixels.clear();
}

bool superpixel::connected(const superpixel &sp1, const superpixel &sp2)
{
    for (const Eigen::Vector2d &pixel1 : sp1.m_pixels)
    for (const Eigen::Vector2d &pixel2 : sp2.m_pixels)
        if (dist_1(pixel1, pixel2) < 2)
            return true;

    return false;
}

double superpixel::dist_1(const Eigen::Vector2d &u, const Eigen::Vector2d &v)
{
    return std::abs(u[0] - v[0]) + std::abs(u[1] - v[1]);
}

} // duho
