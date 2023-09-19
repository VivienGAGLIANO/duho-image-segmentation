#include <iostream>
#include "algorithm.h"

namespace duho
{

    superpixel_generation::superpixel_generation(Eigen::MatrixXd &image, double feature_size, double K, bool normalize) : m_image(image), m_feature_size(feature_size), m_K(K)
    {
        if (normalize)
        {
//            image is in L*a*b* color space, and does not follow gaussian distribution
//            we thus use min-max normalization instead of z-score
//            m_image.colwise() *= Eigen::vectorXd::Constant(m_image.rows(), 1./255.);
            Eigen::VectorXd range = (m_image.colwise().maxCoeff() - m_image.colwise().minCoeff());
            range = 1./range.array();

            m_image.rowwise() -= m_image.colwise().minCoeff();
            m_image *= range.asDiagonal();
        }

        Eigen::MatrixXd image_5d = augmented_matrix(m_image);
        std::cout << image_5d << std::endl;

        // IWASHERE data is normalized, it's K-means o'clock
    }

    superpixel_generation::~superpixel_generation() {}

    std::vector<superpixel> superpixel_generation::generate_superpixels(const Eigen::MatrixXd &image) {
        return std::vector<superpixel>();
    }

    superpixel_generation::augmented_matrix::augmented_matrix(const Eigen::MatrixXd &matrix) : Eigen::MatrixXd(matrix.rows(), matrix.cols()+2), size(std::sqrt(matrix.rows()))
    {
        block(0,0,matrix.rows(),matrix.cols()) = matrix;
        for (size_t ind = 0; ind < rows(); ++ind)
        {
            int i, j;
            ind_to_ij(ind, i, j);
            Eigen::MatrixXd coord(1,2);
            coord << i, j;
            row(ind).tail(2) = coord / size; // same as block(ind,matrix.cols(),1,2) = coord / size;
        }
    }

    void superpixel_generation::augmented_matrix::ind_to_ij(int ind, int &i, int &j)
    {
        // here we make the assumption that the original image is square
        i = ind % (int)size;
        j = ind / (int)size;
    }

    void superpixel_generation::augmented_matrix::ij_to_ind(int i, int j, int &ind)
    {
        ind = j * (int)size + i;
    }

} // duho