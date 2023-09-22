# Limitations
Only works with squared images, because rectangular images are more work :|

The algorithm uses L*a*b* color space, and thus only works on RGB images.

# Additional libraries install

## [png++](https://www.nongnu.org/pngpp/)
C++ wrapper for libpng library, a lightweight PNG image library written in C
- download latest version [here](https://download.savannah.gnu.org/releases/pngpp/)
- follow instructions in INSTALL file, and install the library system-wide

## [eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
Template library for linear algebra, including matrices, vectors, numerical solvers, and related algorithms
- execute `git submodule update --init --recursive` in the root directory of the project

