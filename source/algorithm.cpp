#include <iostream>
#include "algorithm.h"

namespace duho
{

    superpixel_generation::superpixel_generation(Eigen::MatrixXd &image, double feature_size, double K, bool normalize) : m_feature_size(feature_size), m_K(K), m_image(image), m_centers(K)
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
    }

    std::vector<superpixel> superpixel_generation::generate_superpixels()
    {
        augmented_matrix image_5d = augmented_matrix(m_image);

        // Step 1 : Initialize K cluster centers
        const int per_row = static_cast<int>(std::sqrt(m_K));
        const int interval = static_cast<int>(std::sqrt(m_feature_size));

        for (size_t k = 0; k < m_K; ++k)
        {
            int i = (k % per_row)*interval,
                j = (k / per_row)*interval,
                ind;
            image_5d.ij_to_ind(i, j, ind);
            m_centers[k] = image_5d.row(ind);
        }

        // Step 2 : Assign each pixel to the cluster center with the smallest distance

        // Step 3 : Update the cluster centers

        // Step 4 : Repeat steps 2 and 3 until convergence or stopping criterion is met


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