//#include <execution>
#include <iostream>

#include "algorithm.h"

namespace duho
{

    superpixel_generation::superpixel_generation(Eigen::MatrixXd &image, double feature_size, int K, bool normalize) :
        m_feature_size(feature_size),
        m_K(K)/*, m_image(image)*/,
        m_centers(K),
        m_clusters(K),
        m_image_5d(normalize ? normalize_data(image) : image)
    {}

    std::vector<superpixel> superpixel_generation::generate_superpixels()
    {
//        augmented_matrix image_5d = augmented_matrix(m_image);

        // Step 1 : Initialize K cluster centers
        const int per_row = static_cast<int>(std::sqrt(m_K));
        const int interval = static_cast<int>(std::sqrt(m_feature_size));

        // TODO parallelize this
        for (size_t k = 0; k < m_K; ++k)
        {
            int i = (static_cast<int>(k) % per_row)*interval,
                j = (static_cast<int>(k) / per_row)*interval,
                ind;
            m_image_5d.ij_to_ind(i, j, ind);
            m_centers[k] = m_image_5d.row(ind);
        }

        while (m_beta > 0) // TODO implement m_alpha criterion
        {
            // Step 2 : Assign each pixel to the cluster center with the smallest distance
            // TODO parallelize this
            for (size_t ind = 0; ind < m_image_5d.rows(); ++ind)
            {
                Eigen::Vector<double, 5> pixel = m_image_5d.row(ind);
                double d = -1;
                size_t index = 0;
                for (size_t k = 1; k < m_K; ++k)
                {
                    double d_k = weighted_euclidian_distance_squared(pixel, m_centers[k], m_weights);
                    if (d_k < d)
                    {
                        d = d_k;
                        index = k;
                    }
                }
                m_clusters[index].m_pixels.emplace_back(pixel.tail(2));
            }

            // Step 3 : Update the cluster centers by averaging xy coordinates. This (probably) works because superpixels seem to be convex shapes.
            // TODO parallelize this
            for (size_t k = 0; k < m_K; ++k)
            {
                Eigen::Vector<double, 2> center = Eigen::Vector<double, 2>::Zero();
                for (size_t ind = 0; ind < m_clusters[k].m_pixels.size(); ++ind)
                {
                    center += m_clusters[k].m_pixels[ind];
                }
                if (!m_clusters[k].m_pixels.empty())
                    center /= (double)m_clusters[k].m_pixels.size() / m_image_5d.get_size();
                int ind;
                m_image_5d.ij_to_ind(center(0), center(1), ind);


                Eigen::Matrix<double, 5, 1> row = m_image_5d.row(ind);
                m_centers[k] = row;
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

    Eigen::MatrixXd superpixel_generation::clusters_to_image() const
    {
        Eigen::MatrixXd image(m_image_5d.rows(), 3);
        for (size_t k = 0; k < m_K; ++k)
        {
            for (size_t ind = 0; ind < m_clusters[k].m_pixels.size(); ++ind)
            {
                int i = m_clusters[k].m_pixels[ind](0),
                    j = m_clusters[k].m_pixels[ind](1),
                    index;
                m_image_5d.ij_to_ind(i, j, index);

                Eigen::Vector3d row {k, k, k};
                image.row(index) = row;
            }
        }

        return image;
    }

    Eigen::MatrixXd superpixel_generation::normalize_data(Eigen::MatrixXd image)
    {
        Eigen::VectorXd range = (image.colwise().maxCoeff() - image.colwise().minCoeff());
        range = 1./range.array();

        image.rowwise() -= image.colwise().minCoeff();
        image *= range.asDiagonal();

        return image + Eigen::MatrixXd();
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

    void superpixel_generation::augmented_matrix::ind_to_ij(int ind, int &i, int &j) const
    {
        // here we make the assumption that the original image is square
        i = ind % (int)size;
        j = ind / (int)size;
    }

    void superpixel_generation::augmented_matrix::ij_to_ind(int i, int j, int &ind) const
    {
        ind = j * (int)size + i;
    }

    int superpixel_generation::augmented_matrix::get_size() const
    {
        return size;
    }

} // duho