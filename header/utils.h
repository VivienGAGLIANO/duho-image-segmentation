#ifndef DUHO_UTILS_H
#define DUHO_UTILS_H

#include <cmath>
#include <filesystem>
#include <iostream>
#include <png++/png.hpp>

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
    inline void rgb_image_to_lab(png::image<png::rgb_pixel> &image)
    {
        for (size_t y = 0; y < image.get_height(); ++y)
        for (size_t x = 0; x < image.get_width(); ++x)
        {
            // get RGB values
            double r = image[y][x].red;
            double g = image[y][x].green;
            double b = image[y][x].blue;

            // initialize variables for L*a*b* values
            double x_val, y_val, z_val, l_val, a_val, b_val;

            // convert RGB to XYZ
            rgb_to_xyz(r, g, b, x_val, y_val, z_val);

            // convert XYZ to L*a*b*
            xyz_to_lab(x_val, y_val, z_val, l_val, a_val, b_val);

            // store L*a*b* values in the image
            image[y][x].red = static_cast<uint8_t>(l_val);
            image[y][x].green = static_cast<uint8_t>(a_val + 128); // Shift a* to [0, 255]
            image[y][x].blue = static_cast<uint8_t>(b_val + 128);  // Shift b* to [0, 255]
        }
    }

    inline Eigen::MatrixXd normalize_data(Eigen::MatrixXd image)
    {
        Eigen::VectorXd range = (image.colwise().maxCoeff() - image.colwise().minCoeff());
        range = 1. / range.array();

        image.rowwise() -= image.colwise().minCoeff();
        image *= range.asDiagonal();

        return image;
    }

    inline augmented_matrix prepare_data(const png::image<png::rgb_pixel> &image)
    {
        // convert RGB image to L*a*b* color space
        png::image<png::rgb_pixel> lab_image = image;
        rgb_image_to_lab(lab_image);

        // convert image to matrix
        Eigen::MatrixXd matrix = image_to_matrix<png::rgb_pixel>(lab_image);

        // normalize data
        matrix = normalize_data(matrix);

        // add x and y coordinates to matrix for segmentation purposes
        augmented_matrix augmented_matrix(matrix);

        return augmented_matrix;
    }

    // Color hash function from integer to RGB
    inline Eigen::Vector3d color_hash(int i)
    {
        uint8_t r = static_cast<uint8_t>((i * 2654435761U) % 256);
        uint8_t g = static_cast<uint8_t>((i * 2654435761U * 2) % 256);
        uint8_t b = static_cast<uint8_t>((i * 2654435761U * 3) % 256);

        return Eigen::Vector3d(r, g, b);
    }

    inline void write_image(const Eigen::MatrixXd &matrix, const Eigen::Vector2i &dimensions, const std::string &resource_path, const std::string &suffix)
    {
        // select last part of file path as filename
        std::string filename = resource_path.substr(resource_path.find_first_of("/\\")+1);
        filename = "output/" + filename.substr(0,filename.find_last_of('.')) + suffix + ".png";

        auto fp = std::filesystem::path(filename);
        if (!std::filesystem::is_directory(fp.parent_path()))
            std::filesystem::create_directories(fp.parent_path());

        matrix_to_image<png::rgb_pixel>(matrix, dimensions).write(filename);
    }

    inline void test_parameters(double fs_min, double fs_max, const std::string &filename, double step=1.0, bool save=false)
    {
        assert(fs_min > 0 && fs_min <= fs_max);

        // read image and convert to L*a*b* color space
        png::image<png::rgb_pixel> image(filename, png::require_color_space<png::rgb_pixel>());
        duho::augmented_matrix image_matrix = prepare_data(image);

        double alpha = 2;
        uint8_t beta = 10;
        double prev_K = 0;

        for (double fs = fs_min; fs <= fs_max; fs+=step)
        {
            // superpixel generation algorithm
            double K = image.get_width() * image.get_height() / fs;
            K = pow(int(std::sqrt(K)), 2);
            if (prev_K == K) continue;
            prev_K = K;

            // superpixel generation
            duho::superpixel_generation superpixel_generation(image_matrix, fs, K);
            std::vector<duho::superpixel> superpixels = superpixel_generation.generate_superpixels();
            Eigen::MatrixXd clusters_image = superpixel_generation.clusters_to_image();
            if (save) write_image(clusters_image, {image.get_width(), image.get_height()}, filename, "_clusters_" + std::to_string(static_cast<int>(fs)));

            // unseeded region-growing segmentation
            duho::region_growing_segmentation region_growing_segmentation(superpixels, image_matrix);
            std::vector<duho::region_growing_segmentation::region> regions = region_growing_segmentation.segment();
            Eigen::MatrixXd regions_image = region_growing_segmentation.regions_to_image();
            if (save) write_image(regions_image, {image.get_width(), image.get_height()}, filename, "_regions_" + std::to_string(static_cast<int>(fs)));
        }

    }

} // namespace duho

#endif //DUHO_UTILS_H
