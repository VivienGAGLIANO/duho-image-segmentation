#ifndef DUHO_EIGEN_IMAGE_H
#define DUHO_EIGEN_IMAGE_H

#include <execution>

#include "Eigen/Dense"
#include "png++/image.hpp"

namespace duho
{

template <typename T>
inline Eigen::MatrixXd image_to_matrix(const png::image<T> &image);

// Only handles RGB images for now
template <>
inline Eigen::MatrixXd image_to_matrix(const png::image<png::rgb_pixel> &image)
{
    // Check if image is non-empty
    if (image.get_width() == 0 || image.get_height() == 0)
    {
        std::cerr << "Image is empty." << std::endl;
        return {};
    }

    Eigen::MatrixXd matrix(image.get_width()*image.get_height(), 3);

    for (size_t y = 0; y < image.get_height(); ++y)
    for (size_t x = 0; x < image.get_width(); ++x)
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
inline png::image<png::rgb_pixel> matrix_to_image(const Eigen::MatrixXd &matrix, const Eigen::Vector2i &dimensions)
{
    // Check if matrix is non-empty
    if (matrix.rows() == 0 || matrix.cols() == 0)
    {
        std::cerr << "Matrix is empty." << std::endl;
        return {};
    }
    // TODO check matrix has right number of channels

    png::image<png::rgb_pixel> image(dimensions(0), dimensions(1));

    for (size_t y = 0; y < image.get_height(); ++y)
    for (size_t x = 0; x < image.get_width(); ++x)
    {
        png::rgb_pixel pixel;
        pixel.red = matrix(x+image.get_width()*y, 0);
        pixel.green = matrix(x+image.get_width()*y, 1);
        pixel.blue = matrix(x+image.get_width()*y, 2);

        image[y][x] = pixel;
    }

    return image;
}

struct augmented_matrix : public Eigen::MatrixXd
{
public:
    explicit augmented_matrix(const Eigen::MatrixXd &matrix) :
        Eigen::MatrixXd(matrix.rows(), matrix.cols()+2),
        size(std::sqrt(matrix.rows()))
    {
        {
            block(0,0,matrix.rows(),matrix.cols()) = matrix;
            std::vector<size_t> indices = std::vector<size_t>(rows());
            std::iota(indices.begin(), indices.end(), 0);
            std::for_each(std::execution::par_unseq, indices.cbegin(), indices.cend(), [&](size_t ind)
            {
//        for (size_t ind = 0; ind < rows(); ++ind)
//        {
                int i, j;
                ind_to_ij(ind, i, j);
                Eigen::MatrixXd coord(1,2);
                coord << i, j;
                row(ind).tail(2) = coord / size; // same as block(ind,matrix.cols(),1,2) = coord / size;
            });
        }
    }

    inline void ind_to_ij(int ind, int &i, int &j) const
    {
        // here we make the assumption that the original image is square
        i = ind % (int)size;
        j = ind / (int)size;
    }

    inline void ij_to_ind(int i, int j, int &ind) const
    {
        ind = j * (int)size + i;
        if (ind < 0 || ind >= size*size)
            std::cerr << "Error: (" << i << ", " << j << ") index out of bounds." << std::endl;
    }

    int size;
};

} // duho

#endif //DUHO_EIGEN_IMAGE_H
