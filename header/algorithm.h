#ifndef DUHO_ALGORITHM_H
#define DUHO_ALGORITHM_H

#include <vector>

#include "Eigen/Core"
#include "superpixel.h"

namespace duho
{

    class superpixel_generation
    {
    public:
        superpixel_generation()=delete;
        superpixel_generation(Eigen::MatrixXd &image, double feature_size, int K, bool normalize = true);
        ~superpixel_generation()=default;

        std::vector<superpixel> generate_superpixels();
        Eigen::MatrixXd clusters_to_image() const;

        static Eigen::MatrixXd normalize_data(Eigen::MatrixXd image);
        static double weighted_euclidian_distance_squared(const Eigen::Vector<double, 5> &x, const Eigen::Vector<double, 5> &y, const Eigen::Vector<double, 5> &weights);

    private:
        class augmented_matrix : public Eigen::MatrixXd
        {
        public:
            explicit augmented_matrix(const Eigen::MatrixXd &matrix);

            void ind_to_ij(int ind, int &i, int &j) const;
            void ij_to_ind(int i, int j, int &ind) const;

            int get_size() const;

        private:
            int size;
        };

    public: //(this should be private)
        const double m_feature_size;
        const int m_K;
        const double m_alpha = 2;
        uint8_t m_beta = 10;
        Eigen::Vector<double, 5> m_weights = {1, 1, 1, .2, .2};

//        Eigen::MatrixXd &m_image;
        std::vector<Eigen::Matrix<double, 5, 1>> m_centers;
        std::vector<superpixel> m_clusters;

        augmented_matrix m_image_5d;
    };


    class region_growing_segmentation
    {
    public:
        region_growing_segmentation();
        ~region_growing_segmentation();

        static std::vector<superpixel> segment(const Eigen::MatrixXd &image, const std::vector<superpixel> &superpixels);
    };

} // duho

#endif //DUHO_ALGORITHM_H
