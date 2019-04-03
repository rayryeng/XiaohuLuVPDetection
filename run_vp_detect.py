import argparse
from lu_vp_detect.vp_detection import vp_detection

# Set up argument parser + options
parser = argparse.ArgumentParser(
	description='Main script for Lu''s Vanishing Point Algorithm')
parser.add_argument('-i', '--image-path', help='Path to the input image',
	required=True)
parser.add_argument('-lt', '--length-thresh', default=30, type=float,
	help='Minimum line length (in pixels) for detecting lines')
parser.add_argument('-pp', '--principal-point', default=None, nargs=2,
	type=float, help='Principal point of the camera (default is image centre)')
parser.add_argument('-f', '--focal-length', default=1500, type=float,
	help='Focal length of the camera (in pixels)')
parser.add_argument('-d', '--debug', action='store_true',
	help='Turn on debug image mode')
parser.add_argument('-ds', '--debug-show', action='store_true',
	help='Show the debug image in an OpenCV window')
parser.add_argument('-dp', '--debug-path', default=None,
	help='Path for writing the debug image')
args = parser.parse_args()

# Extract command line arguments
input_path = args.image_path
length_thresh = args.length_thresh
principal_point = args.principal_point
focal_length = args.focal_length
debug_mode = args.debug
debug_show = args.debug_show
debug_path = args.debug_path

print('Input path: {}'.format(input_path))
print('Line length threshold: {}'.format(length_thresh))
print('Focal length: {}'.format(focal_length))

# Create object
vpd = vp_detection(length_thresh, principal_point, focal_length)

# Run VP detection algorithm
vps = vpd.find_vps(input_path)
print('Principal point: {}'.format(vpd.principal_point))

# Show VP information
print("The vanishing points in 3D space are: ")
for i, vp in enumerate(vps):
	print("Vanishing Point {:d}: {}".format(i + 1, vp))

vp2D = (vps[:,:2] / vps[:,-1][:,None]) + vpd.principal_point
print("The vanishing points in image coordinates are: ")
for i, vp in enumerate(vp2D):
	print("Vanishing Point {:d}: {}".format(i + 1, vp))

# Extra stuff
if debug_mode or debug_show:
	st = "Creating debug image"
	if debug_show:
		st += " and showing to the screen"
	if debug_path is not None:
		st += "\nAlso writing debug image to: {}".format(debug_path)

	if debug_show or debug_path is not None:
		print(st)
		vpd.create_debug_VP_image(debug_show, debug_path)