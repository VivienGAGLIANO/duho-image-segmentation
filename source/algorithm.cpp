#include "algorithm.h"

namespace duho
{

    superpixel_generation::superpixel_generation(Eigen::MatrixXd &image, double feature_size, double K, bool normalize) : m_image(image), m_feature_size(feature_size), m_K(K)
    {
        if (normalize)
        {
//             image in L*a*b* color space
            m_image(0) /= m_image.col(0).maxCoeff() * 2. - 1.;
            m_image(1) /= m_image.col(1).maxCoeff() * 2. - 1.;
            m_image(2) /= m_image.col(2).maxCoeff() * 2. - 1.;
        }
    }

} // duho