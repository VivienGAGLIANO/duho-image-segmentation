#ifndef DUHO_ALGORITHM_H
#define DUHO_ALGORITHM_H

#include <vector>

#include "Eigen/Core"
#include "superpixel.h"
#include "eigen_image.h"

namespace duho
{
    const Eigen::Vector3d W3 = {1, 1, 1};

    /****************************** Superpixel Generation ******************************/

    class superpixel_generation
    {
    public:
        superpixel_generation()=delete;
        superpixel_generation(augmented_matrix &image, double feature_size, int K);
        ~superpixel_generation()=default;

        std::vector<superpixel> generate_superpixels();
        Eigen::MatrixXd clusters_to_image() const;

        static double weighted_euclidian_distance_squared(const Eigen::Vector<double, 5> &x, const Eigen::Vector<double, 5> &y, const Eigen::Vector<double, 5> &weights);

    private:


    public: //(this should be private)
        const double m_feature_size;
        const int m_K;
        const double m_alpha = 2;
        uint8_t m_beta = 10;
        Eigen::Vector<double, 5> m_weights = {1, 1, 1, .2, .2};

//        Eigen::MatrixXd &m_image;
        std::vector<Eigen::Matrix<double, 5, 1>> m_centers;
        std::vector<superpixel> m_clusters;

        augmented_matrix &m_image_5d;
    };


    /****************************** Region Growing Segmentation ******************************/

    class region_growing_segmentation
    {
    public:
        explicit region_growing_segmentation(const std::vector<superpixel> &superpixels);
        ~region_growing_segmentation()=default;

        std::vector<superpixel> segment(const Eigen::MatrixXd &image, const std::vector<superpixel> &superpixels);

    private:
        class region
        {
        public:
            region()=default;

            void add_superpixel(const superpixel &sp);

            double get_mean() const;
            double get_variance() const;
            double get_size() const;

            static double weighted_distance_squared(const region &r, const superpixel &sp);
            static bool connected(const region &r, const superpixel &sp);

        private:
            std::vector<superpixel> m_superpixels;
            std::vector<double> m_distances;
            Eigen::Vector3d m_mean; // this should always be updated with superpixels currently present in the region
            int q1, q2, q3;
            double iqr;
        };

        std::vector<region> m_regions;
        std::vector<superpixel> m_superpixels;
        std::vector<int> m_unvisited; // stored as indices of superpixels

    };

} // duho

#endif //DUHO_ALGORITHM_H
