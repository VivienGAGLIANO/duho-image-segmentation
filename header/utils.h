#ifndef DUHO_UTILS_H
#define DUHO_UTILS_H

#include <iostream>
#include <png++/png.hpp>
#include <cmath>

namespace duho
{

    // RGB to XYZ conversion
    inline void rgb_to_xyz(double r, double g, double b, double &x, double &y, double &z)
    {
        r /= 255.0;
        g /= 255.0;
        b /= 255.0;

        if (r > 0.04045) r = std::pow((r + 0.055) / 1.055, 2.4);
        else r = r / 12.92;

        if (g > 0.04045) g = std::pow((g + 0.055) / 1.055, 2.4);
        else g = g / 12.92;

        if (b > 0.04045) b = std::pow((b + 0.055) / 1.055, 2.4);
        else b = b / 12.92;

        r *= 100.0;
        g *= 100.0;
        b *= 100.0;

        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
    }

    // XYZ to L*a*b* conversion
    inline void xyz_to_lab(double x, double y, double z, double &l, double &a, double &b)
    {
        x /= 95.047;
        y /= 100.000;
        z /= 108.883;

        if (x > 0.008856) x = std::pow(x, 1.0 / 3.0);
        else x = (x * 903.3 + 16.0) / 116.0;

        if (y > 0.008856) y = std::pow(y, 1.0 / 3.0);
        else y = (y * 903.3 + 16.0) / 116.0;

        if (z > 0.008856) z = std::pow(z, 1.0 / 3.0);
        else z = (z * 903.3 + 16.0) / 116.0;

        l = (116.0 * y) - 16.0;
        a = (x - y) * 500.0;
        b = (y - z) * 200.0;
    }

    // Convert RGB image to L*a*b* color space
    void rgb_image_to_lab(png::image<png::rgb_pixel> &image)
    {
        for (size_t y = 0; y < image.get_height(); ++y)
        for (size_t x = 0; x < image.get_width(); ++x)
        {
            // Get RGB values
            double r = image[y][x].red;
            double g = image[y][x].green;
            double b = image[y][x].blue;

            // Initialize variables for L*a*b* values
            double x_val, y_val, z_val, l_val, a_val, b_val;

            // Convert RGB to XYZ
            rgb_to_xyz(r, g, b, x_val, y_val, z_val);

            // Convert XYZ to L*a*b*
            xyz_to_lab(x_val, y_val, z_val, l_val, a_val, b_val);

            // Store L*a*b* values in the image
            image[y][x].red = static_cast<uint8_t>(l_val);
            image[y][x].green = static_cast<uint8_t>(a_val + 128); // Shift a* to [0, 255]
            image[y][x].blue = static_cast<uint8_t>(b_val + 128);  // Shift b* to [0, 255]
        }
    }

    Eigen::MatrixXd normalize_data(Eigen::MatrixXd image)
    {
        Eigen::VectorXd range = (image.colwise().maxCoeff() - image.colwise().minCoeff());
        range = 1. / range.array();

        image.rowwise() -= image.colwise().minCoeff();
        image *= range.asDiagonal();

        return image;
    }

// Color hash function from integer to RGB


} // namespace duho

#endif //DUHO_UTILS_H
