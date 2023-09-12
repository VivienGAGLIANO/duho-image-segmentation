#include "superpixel.h"

namespace duho
{

superpixel::superpixel() : m_pixels(std::vector<Eigen::Vector2f>()) {}

superpixel::superpixel(const std::vector<Eigen::Vector2f> &pixels) : m_pixels(pixels) {}

superpixel::~superpixel()
{
    m_pixels.clear();
}

bool superpixel::connected(const superpixel &sp1, const superpixel &sp2) {
    return false;
}

} // duho
