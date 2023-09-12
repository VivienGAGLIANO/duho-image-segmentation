#ifndef DUHO_EIGEN_IMAGE_H
#define DUHO_EIGEN_IMAGE_H

#include "Eigen/Dense"
#include "png++/image.hpp"

namespace duho
{

template <typename T>
Eigen::MatrixXd image_to_matrix(const png::image<T> &image);

// Only handles RGB images for now
template <>
Eigen::MatrixXd image_to_matrix(const png::image<png::rgb_pixel> &image)
{
    // Check if image is non-empty
    if (image.get_width() == 0 || image.get_height() == 0)
    {
        std::cerr << "Image is empty." << std::endl;
        return {};
    }

    Eigen::MatrixXd matrix(image.get_width()*image.get_height(), 3);

    for (png::uint_32 y = 0; y < image.get_height(); ++y)
    for (png::uint_32 x = 0; x < image.get_width(); ++x)
    {
        png::rgb_pixel pixel = image[y][x];
        matrix(x+image.get_width()*y, 0) = pixel.red;
        matrix(x+image.get_width()*y, 1) = pixel.green;
        matrix(x+image.get_width()*y, 2) = pixel.blue;
    }

    return matrix;
}

template <typename T>
png::image<T> matrix_to_image(const Eigen::MatrixXd &matrix, const Eigen::Vector2i &dimensions);

template <>
png::image<png::rgb_pixel> matrix_to_image(const Eigen::MatrixXd &matrix, const Eigen::Vector2i &dimensions)
{
    // Check if matrix is non-empty
    if (matrix.rows() == 0 || matrix.cols() == 0)
    {
        std::cerr << "Matrix is empty." << std::endl;
        return {};
    }

    png::image<png::rgb_pixel> image(dimensions(0), dimensions(1));

    for (png::uint_32 y = 0; y < image.get_height(); ++y)
    for (png::uint_32 x = 0; x < image.get_width(); ++x)
    {
        png::rgb_pixel pixel;
        pixel.red = matrix(x+image.get_width()*y, 0);
        pixel.green = matrix(x+image.get_width()*y, 1);
        pixel.blue = matrix(x+image.get_width()*y, 2);

        image[y][x] = pixel;
    }

    return image;
}

} // duho

#endif //DUHO_EIGEN_IMAGE_H
