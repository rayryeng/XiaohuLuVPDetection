# XiaohuLuVPDetection
This is a Python + OpenCV implementation of the Vanishing Point algorithm by Xiaohu Lu et al. - http://xiaohulugo.github.io/papers/Vanishing_Point_Detection_WACV2017.pdf

# Requirements
* Python 3
* OpenCV 3.x
* NumPy

# Setup

Simply run the included `setup.py` file to get it to install onto your local
machine:

```
$ python setup.py build
$ python setup.py install
```

The `lu_vp_detect` package should be installed at this point which will
contain the implementation of the algorithm as well as a command-line test
script to test out the algorithm.

# How to use

The main detection algorithm is written in the `lu_vp_detect` package and is
implemented in the file `vp_detection.py`.  The `vp_detection` class is what
is required and there are two methods of interest in the class:

* `find_vps`: Finds the vanishing points in normalized 3D space
* `create_debug_VP_image`: Creates a debug image for showing which
detected lines in the image align with each vanishing point.

The main parameters for affecting performance are:

* `length_thresh`: The minimum length of the lines detected to find the
vanishing points
* `principal_point`: The principal point of the camera that took the image
(in pixels).  The default is to assume the image centre.
* `focal_length`: The focal length of the camera (in pixels).  The default
is 1500.
* `seed`: An optional integer ID that specifies the seed for reproducibility
as part of the algorithm uses RANSAC.  Default is `None` so no seed is used.

Simply create a `vp_detection` object with the desired parameters and run the
detection algorithm with the desired image.  You can read in the image yourself
or you can provide a path to the image:

```python
from lu_vp_detect.vp_detection import vp_detection
length_thresh = ... # Minimum length of the line in pixels
principal_point = (...,...) # Specify a list or tuple of two coordinates
                            # First value is the x or column coordinate
                            # Second value is the y or row coordinate
focal_length = ... # Specify focal length in pixels
seed = None # Or specify whatever ID you want (integer)

vpd = vp_detection(length_thresh, principal_point, focal_length, seed)

img = '...' # Provide a path to the image
# or you can read in the image yourself
# img = cv2.imread(path_to_image, -1)

# Run detection
vps = vpd.find_vps(img)

# Display vanishing points
print(vps)

# You can also access the VPs directly as a property
# Only works when you run the algorithm first
# vps = vpd.vps

# You can also access the image coordinates for the vanishing points
# Only works when you run the algorithm first
# vps_2D = vpd.vps_2D
```

You can optionally create a debug image that shows which detected lines align
with which vanishing point in the image.  They are colour coded so that each
unique colour corresponds to the lines that provide support for a vanishing
point.  Lines that are black correspond to "outlier" lines, meaning that they
did not contribute any information in calculating any vanishing points.

```python
# Create debug image
vpd.create_debug_VP_image(debug_show=True, debug_path='./path/to/debug.png')
```

The `debug_show` flag will open a OpenCV `imshow` window that will display the
image.  The `debug_path` if it's not set to `None` will write the corresponding
image to file.  Of course, different combinations of the input variables will
allow you to customize what outputs you want to consume or save.

# Command-line test script

The `run_vp_detect.py` file is a command-line script that uses `argparse` to
take in the parameters from the command-line, runs the algorithm and shows
the vanishing points in both 3D normalised space and 2D image coordinate space
in the console.  You can additionally show the debug image and save the debug
image to file by setting the right parameters.

```
$ run_vp_detect -h
usage: run_vp_detect [-h] -i IMAGE_PATH [-lt LENGTH_THRESH]
                     [-pp PRINCIPAL_POINT PRINCIPAL_POINT] [-f FOCAL_LENGTH]
                     [-d] [-ds] [-dp DEBUG_PATH] [-s SEED]

Main script for Lu's Vanishing Point Algorithm

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE_PATH, --image-path IMAGE_PATH
                        Path to the input image
  -lt LENGTH_THRESH, --length-thresh LENGTH_THRESH
                        Minimum line length (in pixels) for detecting lines
  -pp PRINCIPAL_POINT PRINCIPAL_POINT, --principal-point PRINCIPAL_POINT PRINCIPAL_POINT
                        Principal point of the camera (default is image
                        centre)
  -f FOCAL_LENGTH, --focal-length FOCAL_LENGTH
                        Focal length of the camera (in pixels)
  -d, --debug           Turn on debug image mode
  -ds, --debug-show     Show the debug image in an OpenCV window
  -dp DEBUG_PATH, --debug-path DEBUG_PATH
                        Path for writing the debug image
  -s SEED, --seed SEED  Specify random seed for reproducible results
```

Take note that the principal point requires two arguments so you can
separate the `x` and `y` coordinate by a space.

# Example running of the script

A sample image was extracted from [TinyHomes.ie](https://tinyhomes.ie),
specifically this image: [https://www.tinyhomes.ie/wp-content/uploads/2015/09/02.jpg](https://www.tinyhomes.ie/wp-content/uploads/2015/09/02.jpg).
It is saved as `test_image.jpg` in this repository.

The image has been reduced down to 1/4 of the resolution for ease.
The new image has dimensions of `968 x 648`.  The image was originally
taken with a Nikon D40X whose focal length is 18 mm and the CCD width is
15.8 mm.  Therefore, the estimated focal length should be:

`968 x (18 / 15.8) = 1102.79 pixels`

To test the algorithm and reproduce the results below, set the
seed to be 1337 and set the minimum line length to be 60 pixels.  Also,
set the focal length to be what was computed above.  The principal
point is assumed to be the centre of the image.

`$ run_vp_detect -i ./test_image.jpg -f 1102.79 -s 1337 -ds -lt 60`

We get the following output and debug image:

```
Input path: ./test_image.jpg
Seed: 1337
Line length threshold: 60.0
Focal length: 1102.79
Principal point: [484. 324.]
The vanishing points in 3D space are:
Vanishing Point 1: [0.3774699  0.01993015 0.92580736]
Vanishing Point 2: [-0.9259919   0.01616325  0.3771972 ]
Vanishing Point 3: [-0.00744646 -0.99967074  0.0245563 ]

The vanishing points in image coordinates are:
Vanishing Point 1: [933.6292 347.7401]
Vanishing Point 2: [-2223.2698    371.25555]
Vanishing Point 3: [   149.58966 -44569.86   ]
Creating debug image and showing to the screen
```
![](https://i.imgur.com/curdUyC.png)

# License

This code is currently distributed under the MIT license.  Please modify and
use the code as you see fit.  I only ask that you not only include this
license as part of your work but to please acknowledge where you got this
work from!

