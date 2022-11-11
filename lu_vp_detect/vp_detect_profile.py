from lu_vp_detect.vp_detection import LS_ALG, VPDetection
import cProfile
import cv2

def run_(vpd, img):
    for _ in range(60):
        vpd.find_vps(img)


if __name__ == '__main__':

    length_thresh = 20
    principal_point = None
    focal_length = 1102.79
    seed = 1337

    img = '../out2.png'

    vpd = VPDetection(length_thresh=length_thresh, 
                        principal_point=principal_point, 
                        focal_length=focal_length, 
                        seed=seed,
                        line_search_alg=LS_ALG.LSD_WITH_MERGE)
    #vps = vpd.find_vps(img)
    cProfile.run('run_(vpd, img)')