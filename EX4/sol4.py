# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged 

import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
import skimage.color

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite
from scipy.ndimage.filters import convolve
import scipy
import sol4_utils


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """

    # step 1: calculate partial derivatives:
    kernel = np.array([1, 0, -1]).reshape((1, 3))
    Ix_temp = convolve(im, kernel)
    Iy_temp = convolve(im, kernel.T)

    mul = lambda x, y: np.multiply(x, y)
    Ix2 = sol4_utils.blur_spatial(mul(Ix_temp, Ix_temp), 3)
    Iy2 = sol4_utils.blur_spatial(mul(Iy_temp, Iy_temp), 3)
    Ixy = sol4_utils.blur_spatial(mul(Ix_temp, Iy_temp), 3)
    M = np.array([[Ix2, Ixy],
                  [Ixy, Iy2]])

    k = 0.04
    R_temp = np.zeros_like(im)
    for (x, y), value in np.ndenumerate(im):
        m = M[:, :, x, y]
        R_temp[x, y] = (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]) - k * ((m[0, 0] + m[1, 1]) ** 2)

    ret_x, ret_y = non_maximum_suppression(R_temp).nonzero()
    return np.array(np.dstack((ret_y, ret_x))[0], dtype=int)


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    k = 1 + 2 * desc_rad
    deses = []
    for coor_x, coor_y in pos:
        # mappedx, mappedy
        centered_x, centered_y = np.meshgrid(np.arange(coor_x - desc_rad, coor_x + desc_rad + 1),
                                             np.arange(coor_y - desc_rad, coor_y + desc_rad + 1))

        mapped = scipy.ndimage.map_coordinates(im, np.vstack((centered_y.flatten(), centered_x.flatten())),
                                               order=1, prefilter=False)
        des = mapped.reshape((k, k))
        des = des / np.sum(des)
        res = des - np.mean(des)
        fin_des = res / np.linalg.norm(res)
        deses.append(fin_des)
    return np.asarray(deses)


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    pos = spread_out_corners(pyr[0], 7, 7, 16)
    fin_level = pyr[2]
    des = sample_descriptor(fin_level, pos * 0.25, 3)
    return pos, des


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    s = calc_s(desc1, desc2)
    s[s == np.nan] = -1
    before_last = -2
    max_before_last_rows = np.partition(s, before_last, axis=1)
    max_before_last_cols = np.partition(s, before_last, axis=0)
    m1 = max_before_last_rows[:, before_last]
    m2 = max_before_last_cols[before_last, :]

    def find_candidates(i, j):
        i, j = i.astype(int), j.astype(int)
        return (s[i, j] >= m1[i]) & (s[i, j] >= m2[j]) & (s[i, j] > min_score)

    return np.where(np.fromfunction(find_candidates, s.shape))


def calc_s(desc1, desc2):
    n1, n2, k = desc1.shape[0], desc2.shape[0], desc1.shape[1]
    flatten_desc1, flatten_desc2 = desc1.reshape((n1, k ** 2)), desc2.reshape((n2, k ** 2))
    s = np.dot(flatten_desc1, flatten_desc2.T)
    return s


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    # todo: maybe H12 = H12/H12[-1, -1]
    homography = H12 / H12[-1, -1]
    new_pos = np.ones((pos1.shape[0], 3))
    new_pos[0:pos1.shape[0], 0:2] = pos1
    transformed = homography.dot(new_pos.T)  # todo: maybe transposed
    res = np.zeros_like(pos1)
    res[:, 0] = transformed[0] / transformed[2]
    res[:, 1] = transformed[1] / transformed[2]
    return res


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    n = points1.shape[0]
    matches = np.zeros((num_iter, n))
    hs = np.zeros((3, 3, num_iter))
    for k in range(num_iter):
        j = np.random.randint(n, size=1 if translation_only else 2)
        p1, p2 = points1[j], points2[j]

        h = estimate_rigid_transform(p1, p2, translation_only)
        p2_hat = apply_homography(points1, h)

        e = np.linalg.norm(p2_hat - points2, axis=1) ** 2
        matches[k, np.where(e < inlier_tol)] = 1
        hs[:, :, k] = h
    best = np.sum(matches, axis=1).argmax()
    homography = hs[:, :, best]
    return homography / homography[-1, -1], np.where(matches[best] > 0)[0]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    im10, im11, im20, im21 = im1.shape[0], im1.shape[1], im2.shape[0], im2.shape[1]
    bigim = np.empty((np.max([im10, im20]), im11 + im21))
    bigim[:im10, :im11] = im1
    bigim[:im20, im11:im11 + im21] = im2

    plt.imshow(bigim, cmap="gray")

    points2[:, 0] += im11
    for i in range(points1.shape[0]):
        plt.plot([points1[i, 0], points2[i, 0]], [points1[i, 1], points2[i, 1]],
                 mfc="r", c='y' if np.any(inliers == i) else 'b', lw=.4, ms=1, marker="o")
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    H2m = []
    next = np.eye(H_succesive[0].shape[1])  # already 1 :)
    for i in range(m - 1, -1, -1):
        next = np.dot(H_succesive[i], next)
        norm_factor = next[-1, -1]
        H2m.append(next / norm_factor)
    H2m.reverse()
    next = np.eye(H_succesive[0].shape[1])  # already 1 :)
    H2m.append(next)
    # todo: m or m+1?
    for i in range(m, len(H_succesive)):
        h_i_inv = np.linalg.inv(H_succesive[i])
        next = np.dot(next, h_i_inv)
        norm_factor = next[-1, -1]
        H2m.append(next / norm_factor)
    return H2m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    # todo: maybe w-1, h-1
    coords = np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]])

    hm = lambda m: apply_homography(m, homography / homography[-1, -1]).astype(np.int)
    mmin, mmax = lambda x: np.min(x).astype(int), lambda x: np.max(x).astype(int)

    res = hm(coords)
    x, y = res[:, 0], res[:, 1]
    x_min, x_max, y_min, y_max = mmin(x), mmax(x), mmin(y), mmax(y)

    return np.array([[x_min, y_min], [x_max, y_max]])


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    homography /= homography[-1, -1]
    #  index  B
    B = compute_bounding_box(homography, image.shape[1], image.shape[0]).flatten()
    x, y = np.meshgrid(np.arange(B[0], B[2]), np.arange(B[1], B[3]))

    back = apply_homography(np.vstack((x.flatten(), y.flatten())).T, np.linalg.inv(homography))
    back[:, [0, 1]] = back[:, [1, 0]]

    mapped = scipy.ndimage.map_coordinates(image, back.T, order=1, prefilter=False)
    return mapped.reshape((np.abs(B[1] - B[3]).astype(int), np.abs(B[0] - B[2]).astype(int)))


#############################################################################################################
#############################################################################################################
#############################################################################################################

def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in
                      range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies,
                                                                         minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3),
                                  dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()


def _display_side_by_side(im1, im2, f1, f2, i1, i2):
    combined = np.zeros((np.max([im1.shape[0], im2.shape[0]]), im1.shape[1] + im2.shape[1]))

    offset = im1.shape[1]
    combined[:im1.shape[0], :offset] = im1
    combined[:im2.shape[0], offset:offset + im2.shape[1]] = im2

    plt.imshow(combined, cmap="gray")

    selected1 = f1[i1]
    selected2 = f2[i2]

    for i in range(selected1.shape[0]):
        plt.plot([selected1[i, 0], offset + selected2[i, 0]], [selected1[i, 1], selected2[i, 1]],
                 mfc="r", lw=.4, ms=1, marker="o")

    plt.show()


def _match_features_sanity():
    im1 = sol4_utils.read_image("external/oxford1.jpg", 1)
    im2 = sol4_utils.read_image("external/oxford2.jpg", 1)

    pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, 3, 3)
    pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, 3)

    features1, descriptors1 = find_features(pyr1)
    features2, descriptors2 = find_features(pyr2)

    # self1, self2 = match_features(descriptors1, descriptors1, 1)
    # assert np.array_equal(self1, self2), "miss match in indices, expected same indices"

    from1, from2 = match_features(descriptors1, descriptors2, .95)
    _display_side_by_side(im1, im2, features1, features2, from1, from2)


def _display_side_by_side(im1, im2, f1, f2, i1, i2, o1=None, o2=None):
    combined = np.zeros((np.max([im1.shape[0], im2.shape[0]]), im1.shape[1] + im2.shape[1]))

    offset = im1.shape[1]
    combined[:im1.shape[0], :offset] = im1
    combined[:im2.shape[0], offset:offset + im2.shape[1]] = im2

    plt.imshow(combined, cmap="gray")

    if len(f1) == 0 or len(f2) == 0:
        plt.show()
        return

    if o1 is None or o2 is None:
        selected1 = f1[i1]
        selected2 = f2[i2]

        for i in range(selected1.shape[0]):
            plt.plot([selected1[i, 0], offset + selected2[i, 0]], [selected1[i, 1], selected2[i, 1]],
                     mfc="r", lw=.4, ms=1, marker="o")
    else:
        for i in range(i1.shape[0]):
            plt.plot([i1[i, 0], offset + i2[i, 0]], [i1[i, 1], i2[i, 1]],
                     color="yellow", lw=.4, ms=1, marker="o")

        for i in range(o1.shape[0]):
            plt.plot([o1[i, 0], offset + o2[i, 0]], [o1[i, 1], o2[i, 1]],
                     color="blue", lw=.4, ms=1, marker="o")

    plt.show()


GRAY = 1


def _match_features_sanity():
    im1 = sol4_utils.read_image("external/oxford1.jpg", GRAY)
    im2 = sol4_utils.read_image("external/oxford2.jpg", GRAY)

    pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, 3, 3)
    pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, 3)

    features1, descriptors1 = find_features(pyr1)
    features2, descriptors2 = find_features(pyr2)

    self1, self2 = match_features(descriptors1, descriptors1, 1)
    assert np.array_equal(self1, self2), "miss match in indices, expected same indices"

    from1, from2 = match_features(descriptors1, descriptors2, .95)
    _display_side_by_side(im1, im2, features1, features2, from1, from2)


def _ransac_sanity():
    im1 = sol4_utils.read_image("external/oxford1.jpg", GRAY)
    im2 = sol4_utils.read_image("external/oxford2.jpg", GRAY)

    pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, 3, 3)
    pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, 3)

    features1, descriptors1 = find_features(pyr1)
    features2, descriptors2 = find_features(pyr2)

    from1, from2 = match_features(descriptors1, descriptors2, .95)

    points1 = features1[from1]
    points2 = features2[from2]
    h, inliers = ransac_homography(points1, points2, 80, 80)

    o1 = np.delete(points1, inliers, axis=0)
    o2 = np.delete(points2, inliers, axis=0)
    print(inliers.shape[0])
    display_matches(im1, im2, points1, points2, inliers)


def _warp_sanity():
    im1 = sol4_utils.read_image("external/oxford1.jpg", GRAY)
    im2 = sol4_utils.read_image("external/oxford2.jpg", GRAY)

    pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, 3, 3)
    pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, 3)

    features1, descriptors1 = find_features(pyr1)
    features2, descriptors2 = find_features(pyr2)

    from1, from2 = match_features(descriptors1, descriptors2, .95)

    points1 = features1[from1]
    points2 = features2[from2]
    h, inliers = ransac_homography(points1, points2, 80, 80)

    im2_new = warp_channel(im2, np.linalg.inv(h))
    _display_side_by_side(im1, im2_new, [], [], [], [])


if __name__ == '__main__':
    im = sol4_utils.read_image("external/oxford1.jpg", 1)
    _match_features_sanity()
    _ransac_sanity()
    _warp_sanity()
