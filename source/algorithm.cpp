#include <iostream>
#include "algorithm.h"

namespace duho
{

    superpixel_generation::superpixel_generation(Eigen::MatrixXd &image, double feature_size, double K, bool normalize) : m_feature_size(feature_size), m_K(K), m_image(image), m_centers(K), m_clusters(K)
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

        while (m_beta > 0) // TODO implement m_alpha criterion
        {
            // Step 2 : Assign each pixel to the cluster center with the smallest distance
            for (size_t ind = 0; ind < image_5d.rows(); ++ind) {
                Eigen::Vector<double, 5> pixel = image_5d.row(ind);
                double d = -1;
                size_t index = 0;
                for (size_t k = 1; k < m_K; ++k) {
                    double d_k = weighted_euclidian_distance_squared(pixel, m_centers[k], m_weights);
                    if (d_k < d) {
                        d = d_k;
                        index = k;
                    }
                }
                m_clusters[index].m_pixels.emplace_back(pixel.tail(2));
            }

            // Step 3 : Update the cluster centers by averaging xy coordinates
            for (size_t k = 0; k < m_K; ++k) {
                Eigen::Vector<double, 2> center = Eigen::Vector<double, 2>::Zero();
                for (size_t ind = 0; ind < m_clusters[k].m_pixels.size(); ++ind) {
                    center += m_clusters[k].m_pixels[ind];
                }
                center /= m_clusters[k].m_pixels.size();
                int ind;
                image_5d.ij_to_ind(center(0), center(1), ind);
                m_centers[k] = image_5d.row(ind);
            }

            --m_beta;
        }

        // Step 4 : Repeat steps 2 and 3 until convergence or stopping criterion is met


        return m_clusters;
    }

    double superpixel_generation::weighted_euclidian_distance_squared(const Eigen::Vector<double, 5> &x,
                                                              const Eigen::Vector<double, 5> &y,
                                                              const Eigen::Vector<double, 5> &weights)
    {
        return (x-y).transpose() * weights.asDiagonal() * (x-y);
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