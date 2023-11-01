#include <chrono>
#include <execution>
#include <iostream>
#include <mutex>
#include <random>

#include "algorithm.h"
#include "utils.h"

namespace duho
{
    /*********************** Superpixel generation ***********************/

    superpixel_generation::superpixel_generation(augmented_matrix &image, double feature_size, int K) :
        m_feature_size(feature_size),
        m_K(K),
        m_centers(K),
        m_clusters(K, superpixel(std::vector<Eigen::Vector2d>(), image)),
        m_image_5d(image)
    {}

    std::vector<superpixel> superpixel_generation::generate_superpixels()
    {
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();


//        augmented_matrix image_5d = augmented_matrix(m_image);

        // Step 1 : Initialize K cluster centers
        const int per_row = static_cast<int>(std::sqrt(m_K));
        const int interval = static_cast<int>(std::sqrt(m_feature_size));

        std::vector<size_t> indices = std::vector<size_t>(m_K);
        std::iota(indices.begin(), indices.end(), 0);
        std::for_each(std::execution::par_unseq, indices.cbegin(), indices.cend(), [&](size_t k)
        {
//        for (size_t k = 0; k < m_K; ++k)
//        {
            int i = ((static_cast<int>(k) % per_row))*interval,
                j = ((static_cast<int>(k) / per_row))*interval,
                ind;
            m_image_5d.ij_to_ind(i, j, ind);
            m_centers[k] = m_image_5d.row(ind);
        });

        std::vector<std::mutex> mutexes(m_K);

        while (m_beta > 0) // TODO implement m_alpha criterion
        {
            // Clear clusters at the beginning of each iteration
            std::for_each(std::execution::par_unseq, m_clusters.begin(), m_clusters.end(), [&](superpixel &sp)
            {
                sp.m_pixels.clear();
            });

            // Step 2 : Assign each pixel to the cluster center with the smallest distance
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
                m_clusters[index].add_pixel(pixel.tail(2));
            });

            // Step 3 : Update the cluster centers by averaging xy coordinates. This (probably) works because superpixels seem to be convex shapes.
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
                    center /= (double)m_clusters[k].m_pixels.size() / m_image_5d.size;
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
        for (size_t ind = 0; ind < m_clusters[k].m_pixels.size(); ++ind)
        {
            int i = m_clusters[k].m_pixels[ind](0)*m_image_5d.size,
                j = m_clusters[k].m_pixels[ind](1)*m_image_5d.size,
                index;
            m_image_5d.ij_to_ind(i, j, index);

//                image.row(index) = Eigen::Vector3d::Constant(k);
            image.row(index) = color_hash(k);
        }

        return image;// + Eigen::MatrixXd::Constant(image.rows(), image.cols(), 125);
    }


    /*********************** Region growing segmentation ***********************/

    void region_growing_segmentation::region::add_superpixel(const superpixel &sp)
    {
        double distance = weighted_distance_squared(*this, sp);
        auto iterator = std::upper_bound(m_distances.cbegin(), m_distances.cend(), distance);

        // by inserting the superpixel and the distance in the right place we build our vectors sorted by distance to region
        auto index = std::distance(m_distances.cbegin(), iterator);
        m_superpixels.emplace(m_superpixels.begin()+index, sp);
        m_distances.emplace(iterator, distance);

        // update region mean
        m_mean = (m_mean * (m_superpixels.size()-1) + sp.m_mean) / m_superpixels.size();

        // update quantiles
        q1 = m_distances.size() / 4;
        q2 = 2 * m_distances.size() / 4;
        q3 = 3 * m_distances.size() / 4;
        iqr = m_distances[q3] - m_distances[q1];
    }

    bool region_growing_segmentation::region::is_outlier(double distance) const
    {
        if (q1 == q3) return false;

        return distance < m_distances[q1] - 1.5 * iqr || distance > m_distances[q3] + 1.5 * iqr;
    }

    const std::vector<superpixel> region_growing_segmentation::region::get_superpixels() const
    {
        return m_superpixels;
    }

    bool region_growing_segmentation::region::connected(const region_growing_segmentation::region &r, const superpixel &sp)
    {
        for (auto it = r.m_superpixels.cbegin(); it != r.m_superpixels.cend(); ++it)
            if (superpixel::connected(*it, sp))
                return true;

        return false;
    }

    double region_growing_segmentation::region::weighted_distance_squared(const region_growing_segmentation::region &r, const superpixel &sp)
    {
        Eigen::Vector3d diff = r.m_mean - sp.m_mean;
        return diff.transpose() * W3.asDiagonal() * diff;
    }

    region_growing_segmentation::region_growing_segmentation(const std::vector<superpixel> &superpixels, augmented_matrix &image) :
        m_superpixels(superpixels),
        m_unvisited(superpixels.size()),
        m_image_5d(image)
    {
        std::iota(m_unvisited.begin(), m_unvisited.end(), 0);
    }

    std::vector<region_growing_segmentation::region> region_growing_segmentation::segment()
    {
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

        std::list<int> out(1);
        auto random_device = std::mt19937{std::random_device{}()};

        while (!m_unvisited.empty())
        {
            std::sample(m_unvisited.cbegin(), m_unvisited.cend(), out.begin(), 1, random_device);
            int index = out.front();
            m_unvisited.remove(index);
            superpixel sp = m_superpixels[index];

            handle_superpixel(sp);
        }

        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
        std::cout << "Unseeded segmentation execution time : " << duration.count() << "s" << std::endl; // Return the duration in seconds

        return m_regions;
    }

    Eigen::MatrixXd region_growing_segmentation::regions_to_image() const
    {
        Eigen::MatrixXd image(m_image_5d.rows(), 3);
        for (size_t region = 0; region < m_regions.size(); ++region)
        {
            auto superpixels = m_regions[region].get_superpixels();
            for (size_t sp = 0; sp < superpixels.size(); ++sp)
            {
                superpixel superpixel = superpixels[sp];
                for (size_t ind = 0; ind < superpixel.m_pixels.size(); ++ind)
                {
                    int i = superpixel.m_pixels[ind].x() * m_image_5d.size,
                        j = superpixel.m_pixels[ind].y() * m_image_5d.size,
                            index;
                    m_image_5d.ij_to_ind(i, j, index);

        //                image.row(index) = Eigen::Vector3d::Constant(region);
                    image.row(index) = color_hash(region);
                }
            }
        }

        return image;// + Eigen::MatrixXd::Constant(image.rows(), image.cols(), 125);
    }

    void region_growing_segmentation::handle_superpixel(const duho::superpixel &sp)
    {
        if (m_regions.empty())
        {
            region reg;
            reg.add_superpixel(sp);
            m_regions.push_back(reg);

            return;
        }

        double distance = region::weighted_distance_squared(m_regions[0], sp);
        size_t index = 0;
        for (size_t i = 1; i < m_regions.size(); ++i)
        {
            double d = region::weighted_distance_squared(m_regions[i], sp);
            if (d < distance)
            {
                distance = d;
                index = i;
            }
        }

        if (m_regions[index].is_outlier(distance) || !region::connected(m_regions[index], sp))
        {
            region reg;
            reg.add_superpixel(sp);
            m_regions.push_back(reg);
        }

        else
            m_regions[index].add_superpixel(sp);
    }




} // duho