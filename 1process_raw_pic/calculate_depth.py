import pandas as pd
import cv2
import numpy as np
import open3d as o3d
import math


ratio = 12.5/110     # mm/pixel  # get this ratio from : a coin's d = 2r = 25mm, it's pixel_r ~= 110pixels .
R = 6   # ball's r in mm

data = pd.read_excel("/home/ljy/tactile_reconstruction/code/0_creat_data_set/data_df.xls", "Sheet1")

pic = []
param = []

for row in data.index.values:
    pic.append(data.iloc[row,0])
    param.append(data.iloc[row,1])

train_pic_file_root = "/home/ljy/tactile_reconstruction/code/pic_data_set/for_train/"
picture, parameter = pic[0], param[0]
number = f"{picture:>05}"
name = train_pic_file_root + number + ".png"

print(name)

img = cv2.imread(name)
img_temp = cv2.imread(name)

try:
    img.shape
except:
	print("NO such image! check your path")



HW = img.shape
H = HW[0]
W = HW[1]

index_x = np.array(range(H))
index_y = np.array(range(W))
X, Y = np.meshgrid(index_y, index_x)
xy = np.dstack((Y, X))
xy = np.array(xy,dtype=np.float32)
xy *= ratio
# xy = np.array(xy, dtype=np.float32)
# xy_standerd = xy/np.array([640,640])

img_blue, img_green, img_red = cv2.split(img_temp)

parameter_int = list(map(int, parameter.split("-")))
r_pixel, index_x, index_y = parameter_int[0], parameter_int[1], parameter_int[2]

# test the circle's parameters are right
cv2.circle(img, (index_y, index_x), r_pixel, (0, 255, 0), 2)
cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

r = ratio * r_pixel

top_height = 2 * (R - math.sqrt(R**2 - r**2))
# print("top_height = ",top_height)
# print("r_pixel = ",r_pixel,"    index_x  = ",index_x,"    index_y = ",index_y,"  num_circles = ",num_circles)

range = 2
left, right = int(index_y - range*r_pixel), int(index_y + range*r_pixel)
top, down = int(index_x - range*r_pixel), int(index_x + range*r_pixel)

# avoid out of range (480 * 640)
left = 0 if left < 0 else left
right = W if right > W else right
top = 0 if top < 0 else top
down = H if down > H else down

img_g = np.zeros(img.shape[:2], dtype=np.double)    # for cv2

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


# for open3d
points3d = o3d.geometry.PointCloud()

pointsnp = np.dstack((xy, img_g)).reshape(-1, 3)
points3d.points = o3d.utility.Vector3dVector(pointsnp)

o3d.visualization.draw_geometries([points3d])








