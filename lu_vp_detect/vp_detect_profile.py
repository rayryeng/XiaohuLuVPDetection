from vp_detection import VPDetection
import cProfile

def run_(vpd, img):
    for _ in range(10):
        vpd.find_vps(img)


if __name__ == '__main__':

    length_thresh = 20
    principal_point = None
    focal_length = 1102.79
    seed = 1337

    img = '../test_image.jpg'

    vpd = VPDetection(length_thresh, principal_point, focal_length, seed)
    #vps = vpd.find_vps(img)
    cProfile.run('run_(vpd, img)')