# XiaohuLuVPDetection
This is a Python + OpenCV implementation of the Vanishing Point algorithm by Xiaohu Lu et al. - http://xiaohulugo.github.io/papers/Vanishing_Point_Detection_WACV2017.pdf

# Requirements
* Python 3.4 to 3.7 - **Please note that Python 3.8 or higher is not supported at this time as the setup script requires using OpenCV Contrib 4.0.0.21 by default**
* OpenCV Contrib 3.x or 4.0.0.21 - Please note that any version higher than 4.0.0.21 does not have the LSD detection algorithm included, which is a key component for this method to work.  If you have a higher version of OpenCV installed, this will not work.
* NumPy

# Setup

## Method #1 - Cloning this repo and installing locally

Simply clone this repo, then run the included `setup.py` file to get it to install onto your local machine:

```
$ python setup.py build
$ python setup.py install
```

## Method #2 - Through PyPI

This project is now available through PyPI: https://pypi.org/project/lu-vp-detect/

If don't want to clone this repo, you can simply use `pip install` to install this project on your system.

```
$ pip install lu-vp-detect
```

## After installation

The `lu_vp_detect` package should be installed at this point which will
contain the implementation of the algorithm as well as a command-line test
script to test out the algorithm.

# How to use

The main detection algorithm is written in the `lu_vp_detect` package and is
implemented in the file `vp_detection.py`.  The `VPDetection` class is what
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

Simply create a `VPDetection` object with the desired parameters and run the
detection algorithm with the desired image.  You can read in the image yourself
or you can provide a path to the image.  Note that the returned vanishing
points will be a 3 x 3 NumPy array such that the first row corresponds to the
vanishing point appearing to the right of the image, the second row
corresponds to the vanishing point appearing to the left of the image and the
last row corresponding to the vertical vanishing point:

```python
from lu_vp_detect import VPDetection
length_thresh = ... # Minimum length of the line in pixels
principal_point = (...,...) # Specify a list or tuple of two coordinates
                            # First value is the x or column coordinate
                            # Second value is the y or row coordinate
focal_length = ... # Specify focal length in pixels
seed = None # Or specify whatever ID you want (integer)

vpd = VPDetection(length_thresh, principal_point, focal_length, seed)

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
vpd.create_debug_VP_image(show_image=True, save_image='./path/to/debug.png')
```

The `show_image` flag will open a OpenCV `imshow` window that will display the
image.  The `save_image` if it's not set to `None` will write the corresponding
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

For completeness, to do this programmatically:

```python
from lu_vp_detect import VPDetection
length_thresh = 60
principal_point = None
focal_length = 1102.79
seed = 1337

img = './test_image.jpg'

vpd = VPDetection(length_thresh, principal_point, focal_length, seed)
vps = vpd.find_vps(img)
print(vps)
```

We get the following output and debug image:

```
Input path: ./test_image.jpg
Seed: 1337
Line length threshold: 60.0
Focal length: 1102.79
Principal point: [484. 324.]
The vanishing points in 3D space are:
Vanishing Point 1: [0.3774699  0.01993015 0.92580736]
Vanishing Point 2: [-0.9260219   0.00812403  0.37738246]
Vanishing Point 3: [-0.         -0.9997684   0.02152233]

The vanishing points in image coordinates are:
Vanishing Point 1: [933.6292 347.7401]
Vanishing Point 2: [-2222.0286   347.7401]
Vanishing Point 3: [   484.    -50903.473]
Creating debug image and showing to the screen
```
![](https://i.imgur.com/svI8tSC.png)

# License

This code is currently distributed under the MIT license.  Please modify and
use the code as you see fit.  I only ask that you not only include this
license as part of your work but to please acknowledge where you got this
work from!

