from sol4 import *
from sol4_utils import *
import time

if __name__ == '__main__':
    t = time.time()
    im1 = read_image('external\oxford1.jpg', 1)
    im2 = read_image('external\oxford2.jpg', 1)
    p1, d1 = find_features(build_gaussian_pyramid(im1, 3, 3)[0])
    p2, d2 = find_features(build_gaussian_pyramid(im2, 3, 3)[0])
    m1, m2 = match_features(d1, d2, 0.9)
    mp1, mp2 = p1[m1], p2[m2]
    H, inliers = ransac_homography(mp1, mp2, 50, 6)
    display_matches(im1, im2, mp1, mp2, inliers)
    im_rgb = read_image('external\oxford1.jpg', 2)
    warped = warp_image(im_rgb, H)
    print(time.time() - t)
    print(warped.shape)
    print(H)
    plt.imshow(warped)
    plt.show()
    Hs = [np.eye(3) for i in range(2)]
    aHs = accumulate_homographies(Hs, 1)
    pass
