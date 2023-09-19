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
        superpixel_generation(Eigen::MatrixXd &image, double feature_size, double K, bool normalize = true);
        ~superpixel_generation();

        static std::vector<superpixel> generate_superpixels(const Eigen::MatrixXd &image);

    public: //(this should be private)
        Eigen::MatrixXd &m_image;
        double m_feature_size;
        double m_K;
        double m_alpha = 2;
        uint8_t m_beta = 10;

    private:
    class augmented_matrix : public Eigen::MatrixXd
    {
    public:
        explicit augmented_matrix(const Eigen::MatrixXd &matrix);

        void ind_to_ij(int ind, int &i, int &j);
        void ij_to_ind(int i, int j, int &ind);

    private:
        const int size;
    };
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
