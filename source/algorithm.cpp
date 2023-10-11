#include <chrono>
#include <execution>
#include <iostream>
#include <mutex>
#include <random>

#include "algorithm.h"

namespace duho
{
    /*********************** Superpixel generation ***********************/

    superpixel_generation::superpixel_generation(Eigen::MatrixXd &image, double feature_size, int K, bool normalize) :
        m_feature_size(feature_size),
        m_K(K)/*, m_image(image)*/,
        m_centers(K),
        m_clusters(K),
        m_image_5d(normalize ? normalize_data(image) : image)
    {}

    std::vector<superpixel> superpixel_generation::generate_superpixels()
    {
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();


//        augmented_matrix image_5d = augmented_matrix(m_image);

        // Step 1 : Initialize K cluster centers
        const int per_row = static_cast<int>(std::sqrt(m_K));
        const int interval = static_cast<int>(std::sqrt(m_feature_size));

        // TODO parallelize this
        std::vector<size_t> indices = std::vector<size_t>(m_K);
        std::iota(indices.begin(), indices.end(), 0);
        std::for_each(std::execution::par_unseq, indices.cbegin(), indices.cend(), [&](size_t k)
        {
//        for (size_t k = 0; k < m_K; ++k)
//        {
            // TODO fix this, it sometimes crashes because it generates out of bound indices
            int i = ((static_cast<int>(k) % per_row)+0.5)*interval,
                j = ((static_cast<int>(k) / per_row)+0.5)*interval,
                ind;
            m_image_5d.ij_to_ind(i, j, ind);
            m_centers[k] = m_image_5d.row(ind);
        });

        std::vector<std::mutex> mutexes(m_K);

        while (m_beta > 0) // TODO implement m_alpha criterion
        {
            // Step 2 : Assign each pixel to the cluster center with the smallest distance
            // TODO parallelize this
            indices.resize(m_image_5d.rows());
            std::iota(indices.begin(), indices.end(), 0);
            std::for_each(std::execution::par_unseq, indices.cbegin(), indices.cend(), [&](size_t ind)
            {
//            for (size_t ind = 0; ind < m_image_5d.rows(); ++ind)
//            {
                Eigen::Vector<double, 5> pixel = m_image_5d.row(ind);
                double d = weighted_euclidian_distance_squared(pixel, m_centers[0], m_weights);
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
                std::lock_guard<std::mutex> lock(mutexes[index]);
                m_clusters[index].m_pixels.emplace_back(pixel.tail(2));
            });

            // Step 3 : Update the cluster centers by averaging xy coordinates. This (probably) works because superpixels seem to be convex shapes.
            // TODO parallelize this
            indices.resize(m_K);
            std::iota(indices.begin(), indices.end(), 0);
            std::for_each(std::execution::par_unseq, indices.cbegin(), indices.cend(), [&](size_t k)
            {
//            for (size_t k = 0; k < m_K; ++k)
//            {
                Eigen::Vector<double, 2> center = Eigen::Vector<double, 2>::Zero();
                for (size_t ind = 0; ind < m_clusters[k].m_pixels.size(); ++ind)
                {
                    center += m_clusters[k].m_pixels[ind];
                }
                if (!m_clusters[k].m_pixels.empty())
                    center /= (double)m_clusters[k].m_pixels.size() / m_image_5d.get_size();
                int ind;
                m_image_5d.ij_to_ind(center(0), center(1), ind);

                m_centers[k] = m_image_5d.row(ind);
            });

            --m_beta;
        }

        // Step 4 : Repeat steps 2 and 3 until convergence or stopping criterion is met

        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
        std::cout << "Superpixel generation execution time : " << duration.count() << "s" << std::endl; // Return the duration in seconds

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
                int i = m_clusters[k].m_pixels[ind](0)*m_image_5d.get_size(),
                    j = m_clusters[k].m_pixels[ind](1)*m_image_5d.get_size(),
                    index;
                m_image_5d.ij_to_ind(i, j, index);

                image.row(index) = Eigen::Vector3d::Constant(k);
            }
        }

        return image;// + Eigen::MatrixXd::Constant(image.rows(), image.cols(), 125);
    }

    Eigen::MatrixXd superpixel_generation::normalize_data(Eigen::MatrixXd image)
    {
        Eigen::VectorXd range = (image.colwise().maxCoeff() - image.colwise().minCoeff());
        range = 1./range.array();

        image.rowwise() -= image.colwise().minCoeff();
        image *= range.asDiagonal();

        return image;
    }

    superpixel_generation::augmented_matrix::augmented_matrix(const Eigen::MatrixXd &matrix) :
        Eigen::MatrixXd(matrix.rows(), matrix.cols()+2),
        size(std::sqrt(matrix.rows()))
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


    /*********************** Region growing segmentation ***********************/

    std::vector<superpixel> region_growing_segmentation::segment(const Eigen::MatrixXd &image, const std::vector<superpixel> &superpixels)
    {
        std::vector<int> out;
        std::sample(m_unvisited.cbegin(), m_unvisited.cend(), out.begin(), 1, std::mt19937{std::random_device{}()});


        return std::vector<superpixel>();
    }

    region_growing_segmentation::region_growing_segmentation(const std::vector<superpixel> &superpixels) :
        m_superpixels(superpixels),
        m_unvisited(superpixels.size())
    {}



    void region_growing_segmentation::region::add_superpixel(const superpixel &sp)
    {

    }

    bool region_growing_segmentation::region::connected(const region_growing_segmentation::region &r, const superpixel &sp)
    {
        for (auto it = r.m_superpixels.cbegin(); it != r.m_superpixels.cend(); ++it)
            if (superpixel::connected(*it, sp))
                return true;

        return false;
    }


} // duho