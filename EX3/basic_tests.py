from sol3 import *



if __name__ == '__main__':
    file_name = "monkey.jpg"
    org_im = read_image(file_name, 1)
    max_levels = 3
    filter_size = 4

    # 3.1
    g_pyr, filter_vec = build_gaussian_pyramid(org_im, max_levels, filter_size)
    l_pyr, filter_vec = build_laplacian_pyramid(org_im, max_levels, filter_size)

    print("\n***** build_gaussian_pyramid & build_laplacian_pyramid basic tests *****\n")
    flag = True
    if len(l_pyr) != len(g_pyr):
        print("*** error *** \n problem in the lengths, gaussian:", len(g_pyr), "laplacian:", len(l_pyr), "\n")
        flag = False

    for i in range(len(g_pyr)):
        if g_pyr[i].shape != l_pyr[i].shape:
            print("*** error *** \nproblem in the dimensions in level:", i)
            print("gaussian pyr dimensions:", g_pyr[i].shape)
            print("laplacian pyr dimensions:", l_pyr[i].shape, "\n")
            flag = False

    if flag:
        print("\n*** build_gaussian_pyramid, build_laplacian_pyramid passed basic test :) ***\n")

    # 3.2
    coeff = [1, 1, 1, 1]
    img = laplacian_to_image(l_pyr, filter_vec, coeff)

    print("\n***** laplacian_to_image basic tests *****\n")
    flag = True
    sub = img-org_im
    for row in range(len(org_im)):
        for pixel in range(len(org_im[row])):
            if abs(sub[row][pixel]) > (10**-12):
                print("*** error *** difference:", sub[row][pixel])
                flag = False
    if flag:
        print("*** laplacian_to_image passed basic test :) ***\n")

    # 3.3
    levels = len(g_pyr)

    g_res = render_pyramid(g_pyr, levels)
    display_pyramid(g_pyr, levels)

    l_res = render_pyramid(l_pyr, levels)
    display_pyramid(l_pyr, levels)

    print("\n***** render_pyramid & display_pyramid basic tests *****\n")
    print("*** if they look ok - laplacian_to_image passed basic test :) ***\n")

    # 4
    print("\n***** pyramid_blending basic tests *****\n")
    file_name1 = "pizza.jpg"
    file_name2 = "turtle.jpg"
    mask_name = "mask1.jpg"
    im1 = read_image(file_name1, 1)
    im2 = read_image(file_name2, 1)
    mask = make_me_boolean(read_image(mask_name, 1))
    filter_size_im = 5
    filter_size_mask = 5
    im_blend = pyramid_blending(im2, im1, mask, max_levels, filter_size_im, filter_size_mask)
    plt.imshow(im_blend, cmap='gray')
    plt.title("pyramid_blending - basic test")
    plt.show()
    print("*** if it looks ok - pyramid_blending passed basic test :) ***\n")

    # 4.1
    im1, im2, mask, im_blend = blending_example1()
    # im1, im2, mask, im_blend = blending_example2()



