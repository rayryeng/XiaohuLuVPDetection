from lu_vp_detect import VPDetection
import os
import cv2
length_thresh = 60
principal_point = None
focal_length = 1500
seed = 1337

size = 1

outpath = "lu_vp_detect/results/"
inpath = "lu_vp_detect/data/"

for image_path in os.listdir(inpath):
    if(image_path.endswith(".jpg")):
        img = os.path.join(inpath, image_path)
        vpd = VPDetection(length_thresh, principal_point, focal_length, seed)
        vpd.find_vps(img)
        vps2d = vpd.vps_2D
        img = vpd.create_debug_VP_image(show_vp=True)
        cv2.imwrite(outpath + image_path, img)

# for i in range(size):
#     img = "test_image.jpg"
#     vpd = VPDetection(length_thresh, principal_point, focal_length, seed)
#     vpd.find_vps(img)
#     vps2d = vpd.vps_2D
#     img = vpd.create_debug_VP_image(show_vp=True)
#     cv2.imwrite("lu_vp_detect/results/result.jpg", img)