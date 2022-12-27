import cv2


ratio = 12.5/110
R = 6
file_name = "/home/ljy/tactile_reconstruction/code/pic_data_set/for_train/00000.png"
img = cv2.imread(filename=file_name)

b, g, r = cv2.split(img)

cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)

cv2.waitKey(0)
cv2.destroyAllWindows()
