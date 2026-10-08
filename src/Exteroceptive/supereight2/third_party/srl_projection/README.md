README                        {#mainpage}
======


### License ###

The 3-clause BSD license (see file LICENSE) applies.

### How do I get set up? ###

This is a pure cmake project.

You will need to install the following dependencies,

* CMake,

        sudo apt-get install cmake

* Eigen3,

        sudo apt-get install libeigen3-dev


* OpenCV 2.4-3.0: follow the instructions on http://opencv.org/ or install
  via

        sudo apt-get install libopencv-dev

* gtest

        sudo apt-get install libgtest-dev

### Building the project ###

To change the cmake build type for the whole project use:

    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j

### Running the tests ###

To run the gtests after building, do:

    make test

### Installing ###

To install system-wide after building, do:

    sudo make install

### Using in your project ###

To use in your CMake project, after installing system-wide, put the following in your
CMakeLists.txt:

```
find_package(SRLProjection)
target_link_libraries(your_target SRL::Projection)
```



### Contribution guidelines ###

* Programming guidelines: please follow the SRL coding standard.

* Writing tests: please write unit tests (gtest).

* Code review: please create a pull request for all changes proposed. The pull
  request will be reviewed by an admin before merging.

