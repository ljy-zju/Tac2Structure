import time
import cv2
import copy
import _thread

number = 0
key = 0
cap = cv2.VideoCapture(0)

frame = cap.read()
display = copy.deepcopy(frame)
key = 1


def get_picture():
    global frame
    global display
    global key
    while True:
        _, frame = cap.read()
        display = copy.deepcopy(frame)
        r = 160
        x = 320
        y = 240

        cv2.circle(display, center=(x, y), radius=r,
                   color=(255, 0, 0), thickness=2)
        cv2.rectangle(display, (x-r, y-r), (x+r, y+r), (0, 255, 0), 2)
        cv2.imshow("rbg", display)
        key = cv2.waitKey(10)
        # cv2.destroyAllWindows()


# _thread.start_new_thread(get_picture, ())

# while True:
#     if key == 113:
#         while True:
#             time.sleep(10)
#             cv2.imwrite("/home/ljy/python_ws/design_underground2/pic_data_set/for_train/" + f"{number:>05}" + ".png", frame)
#             number += 1
#             print(f"save pic number : {number} !!!")
#             cv2.destroyAllWindows()

while True:
    _, frame = cap.read()

    display = copy.deepcopy(frame)
    r = 160
    x = 320
    y = 240

    cv2.circle(display, center=(x, y), radius=r,
               color=(255, 0, 0), thickness=2)
    cv2.rectangle(display, (x-r, y-r), (x+r, y+r), (0, 255, 0), 2)

    cv2.imshow("rbg", display)

    key = cv2.waitKey(5)

    if key == 113:
        cv2.imwrite("/home/ljy/tactile_reconstruction/code/pic_data_set/new_for_test/" +
                    f"{number:>05}" + ".png", frame)
        number += 1
        cv2.destroyAllWindows()
        print(f"save pic number : {number} !!!")
    else:
        continue
