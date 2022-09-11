from os import listdir
from os.path import join
import cv2
import numpy as np

skin_db = [0] * 17000000
n_skin_db = [0] * 17000000


def get_files_from_path(i_path: str):
    img_files = listdir(i_path)
    img_file_paths = [join(i_path, f) for f in img_files]
    return img_file_paths


def get_img_data(skin_img_list: list, non_skin_img_list: list):
    total_images = len(skin_img_list)
    print(total_images)
    for image_num in range(total_images):
        print("Image num = " + str(image_num))
        # img_skin = Image.open(skin_img_list[image_num])
        img_skin = cv2.imread(skin_img_list[image_num])
        img_non_skin = cv2.imread(non_skin_img_list[image_num])
        im_h, im_w, _ = img_skin.shape
        print (im_h, im_w)
        for h in range(im_h):
            for w in range(im_w):
                sb, sg, sr = img_skin[h, w]
                if sb == 255 and sg == 255 and sr == 255:
                    nb, ng, nr = img_non_skin[h, w]
                    n_skin_idx = nb * 255 * 255 + ng*255 + nr
                    n_skin_db[n_skin_idx] += 1
                else:
                    skin_idx = sb * 255 * 255 + sg * 255 + sr
                    skin_db[skin_idx] += 1


def convert_img(img_test, skin_np_arr, nskin_np_arr):
    img_tt = cv2.imread(img_test)
    im_h, im_w, _ = img_tt.shape
    for h in range(im_h):
        for w in range(im_w):
            b, g, r = img_tt[h, w]
            idx = b*255*255 + g*255 + r
            if nskin_np_arr[idx] == 0:
                if skin_np_arr[idx] == 0:
                    continue
                else:
                    img_tt[h, w] = (0, 0, 0)
            else:
                ratio = skin_np_arr[idx]/nskin_np_arr[idx]
                if ratio> 1:
                    img_tt[h, w] = (0, 0, 0)
    cv2.imwrite("Result_"+ img_test , img_tt)
    pass
if __name__ == '__main__':
    '''skin_img_path = "D:\ibtd\ibtd\Mask"
    skin_images = get_files_from_path(skin_img_path)
    non_skin_path = "D:\ibtd\ibtd\Test"
    non_skin_images = get_files_from_path(non_skin_path)
    get_img_data(skin_images, non_skin_images)

    numpy_skin = np.array(skin_db)
    numpy_nskin = np.array(n_skin_db)

    np.save('skin', numpy_skin)
    np.save('nskin', numpy_nskin)
    
    skin_np = np.load("skin.npy")
    nskin_np = np.load("nskin.npy")

    total_skin_cnt = np.count_nonzero(skin_np)
    skin_np_prob = skin_np / total_skin_cnt

    total_nskin_cnt = np.count_nonzero(nskin_np)
    nskin_np_prob = nskin_np / total_nskin_cnt

    np.save('skin_prob', skin_np_prob)
    np.save('nskin_prob', nskin_np_prob)
    '''
    skin_np_prob = np.load("skin_prob.npy")
    nskin_np_prob = np.load("nskin_prob.npy")

    sample_img = "multiple_person.PNG"
    convert_img(sample_img, skin_np_prob, nskin_np_prob)


