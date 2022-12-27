import pandas as pd
import cv2
import numpy as np
import open3d as o3d
import math
import copy


ratio = 12.5/110     # mm/pixel  # get this ratio from : a coin's d = 2r = 25mm, it's pixel_r ~= 110pixels .
R = 6   # ball's r in mm
centry_x = 240
centry_y = 320
delete_r = 160
circle_trans_x = 80
circle_trans_y = 160


def get_xy_matrix(H, W):
    index_x = np.array(range(H))
    index_y = np.array(range(W))
    X, Y = np.meshgrid(index_y, index_x)
    xy = np.dstack((Y, X))
    xy = np.array(xy,dtype=np.float32)
    xy *= ratio
    return xy


data = pd.read_excel("/home/ljy/tactile_reconstruction/code/0_creat_data_set/data_df_train.xls", "Sheet1")
pic = []
param = []

for row in data.index.values:
    pic.append(data.iloc[row, 0])
    param.append(data.iloc[row, 1])

count = 0
data_write_in_xls = {}
data_write_in_xls["x"] = []
data_write_in_xls["y"] = []
data_write_in_xls["r"] = []
data_write_in_xls["g"] = []
data_write_in_xls["b"] = []
data_write_in_xls["gx"] = []
data_write_in_xls["gy"] = []

writer = pd.ExcelWriter("/home/ljy/tactile_reconstruction/code/2generate_train_test_dataset/train_circles_vpaper.xls")
train_pic_file_root = "/home/ljy/tactile_reconstruction/code/pic_data_set/for_train/"

for picture, parameter in zip(pic,param):
    number = f"{picture:>05}"
    name = train_pic_file_root + number + ".png"
    # print(name)
    img = cv2.imread(name)
    img_cropped = img[centry_x - delete_r:centry_x + delete_r, centry_y - delete_r:centry_y + delete_r]
    img_cropped_temp = copy.deepcopy(img_cropped)

    HW = img_cropped.shape
    H, W = HW[0], HW[1]
    # xy = get_xy_matrix(H, W)

    img_blue, img_green, img_red = cv2.split(img_cropped_temp)

    parameter_int = list(map(int, parameter.split("-")))
    r_pixel, index_x, index_y = parameter_int[0], parameter_int[1] - circle_trans_x, parameter_int[2] - circle_trans_y

    # test the circle is right!
    # cv2.circle(img_cropped, (index_y, index_x), r_pixel, (0, 255, 0), 2)
    # cv2.imshow("test", img_cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    r = ratio * r_pixel
    top_height = 2 * (R - math.sqrt(R**2 - r**2))
    # print("top_height = ",top_height)
    # print("r_pixel = ",r_pixel,"    index_x  = ",index_x,"    index_y = ",index_y,"  num_circles = ",num_circles)

    range = 2
    left, right = int(index_y - range*r_pixel), int(index_y + range*r_pixel)
    top, down = int(index_x - range*r_pixel), int(index_x + range*r_pixel)

    # avoid out of range (320 * 320)
    left = 0 if left < 0 else left
    right = W if right > W else right
    top = 0 if top < 0 else top
    down = H if down > H else down

    img_g = np.zeros(img_cropped.shape[:2], dtype=np.double)    # for cv2

    for i in range(top, down):
        for j in range(left, right):
            length = math.sqrt((i - index_x)**2 + (j - index_y)**2)
            if length > 2*r_pixel:
                continue
            elif length > r_pixel:
                length_real = (2*r_pixel - length) * ratio
                img_g[i, j] = R - math.sqrt(R**2 - length_real**2)
                pass
            else:
                length_real = length * ratio
                img_g[i, j] = (top_height - (R - math.sqrt(R**2 - length_real**2)))


    # re calc fanwei and left\right\top\down
    fanwei = 1.5
    left, right = int(index_y - fanwei*r_pixel), int(index_y + fanwei*r_pixel)
    top, down = int(index_x - fanwei*r_pixel), int(index_x + fanwei*r_pixel)

    # avoid out of range (320 * 320)
    left = 0 if left < 0 else left
    right = W if right > W else right
    top = 0 if top < 0 else top
    down = H if down > H else down


    for i in range(top+1, down-1, 2):
        for j in range(left+1, right-1, 2):
            gx = img_g[i+1, j] - img_g[i, j]
            gy = img_g[i, j+1] - img_g[i, j]
            data_write_in_xls["x"].append(i)
            data_write_in_xls["y"].append(j)
            data_write_in_xls["r"].append(img_red[i, j])
            data_write_in_xls["g"].append(img_green[i, j])
            data_write_in_xls["b"].append(img_blue[i, j])
            data_write_in_xls["gx"].append(gx)
            data_write_in_xls["gy"].append(gy)

    df = pd.DataFrame(data_write_in_xls)
    df.to_excel(writer, sheet_name=number, index=False)
    data_write_in_xls = {}
    data_write_in_xls["x"] = []
    data_write_in_xls["y"] = []
    data_write_in_xls["r"] = []
    data_write_in_xls["g"] = []
    data_write_in_xls["b"] = []
    data_write_in_xls["gx"] = []
    data_write_in_xls["gy"] = []
    print(count)
    count += 1

# add empty data set
picture = 60
number = f"{picture:>05}"
name = train_pic_file_root + number + ".png"
img = cv2.imread(name)
img_cropped = img[centry_x - delete_r:centry_x + delete_r, centry_y - delete_r:centry_y + delete_r]
img_cropped_temp = copy.deepcopy(img_cropped)

HW = img_cropped.shape
H, W = HW[0], HW[1]

img_blue, img_green, img_red = cv2.split(img_cropped_temp)

for i in range(0, H, 2):
    for j in range(0, W, 2):
        data_write_in_xls["x"].append(i)
        data_write_in_xls["y"].append(j)
        data_write_in_xls["r"].append(img_red[i, j])
        data_write_in_xls["g"].append(img_green[i, j])
        data_write_in_xls["b"].append(img_blue[i, j])
        data_write_in_xls["gx"].append(0.0)
        data_write_in_xls["gy"].append(0.0)

df = pd.DataFrame(data_write_in_xls)
df.to_excel(writer, sheet_name=number, index=False)
print(count)

writer.save()




