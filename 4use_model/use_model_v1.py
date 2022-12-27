# v1 : real-time visualize!!
# for gpu model and new dataset trained model
import cv2
import copy
import numpy as np
import open3d as o3d

import torch
import torch.nn as nn

from poisson_reconstruct import poisson_reconstruct


ratio = 12.5/110     # mm/pixel  # get this ratio from : a coin's d = 2r = 25mm, it's pixel_r ~= 110pixels .
R = 6   # ball's r in mm
centry_x = 240
centry_y = 320
delete_r = 160
circle_trans_x = 80
circle_trans_y = 160


class MyMLP(nn.Module):
    # 5 ~ 32 ~ 32 ~ 32 ~ 2
    def __init__(self):
        super(MyMLP, self).__init__()
        hidden = 32
        self.fc1 = nn.Linear(5, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, 2)

        self.drop_out = nn.Dropout(0.02)

    def forward(self, x):
        x = x.view(-1, 5)
        x = self.drop_out(torch.tanh(self.fc1(x)))
        x = self.drop_out(torch.tanh(self.fc2(x)))
        x = self.drop_out(torch.tanh(self.fc3(x)))
        x = self.drop_out(torch.tanh(self.fc4(x)))

        x = self.fc5(x)
        return x


model_test = MyMLP()
model_test = MyMLP().cuda()
model_test.load_state_dict(torch.load("/home/ljy/tactile_reconstruction/code/3train_model/good_model/196/MLP6.pkl"))
model_test.eval()


def get_depth(img):
    img_coin = copy.deepcopy(img)
    img_coin = cv2.cvtColor(img_coin, cv2.COLOR_BGR2RGB)
    gray_coin = cv2.cvtColor(img_coin, cv2.COLOR_RGB2GRAY)
    HW = gray_coin.shape  # 480 640
    H = HW[0]
    W = HW[1]

    index_x = np.array(range(H))
    index_y = np.array(range(W))
    X, Y = np.meshgrid(index_y, index_x)
    xy = np.dstack((Y, X))

    gray_coin_temp = np.array(gray_coin, dtype=np.float32)

    xy = np.array(xy, dtype=np.float32)
    img_coin_np = np.array(img_coin, dtype=np.float32)
    xy_standerd = xy / 320
    img_coin_np = img_coin_np / 255

    vector = torch.tensor(np.dstack((xy_standerd, img_coin_np)), dtype=torch.float32).reshape(-1, 5).cuda()

    gx_gy = model_test(vector).cpu()
    gx = gx_gy[:, 0].reshape(H, W).detach().numpy()
    gy = gx_gy[:, 1].reshape(H, W).detach().numpy()

    boundary_img = np.zeros_like(gray_coin_temp)
    img_rec_np = poisson_reconstruct(gx, gy, boundary_img)


    xy *= ratio
    np_points = np.dstack((xy, img_rec_np)).reshape(-1, 3)

    return np_points


def main():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    points3d = o3d.geometry.PointCloud()
    vis.add_geometry(points3d)
    to_reset = True

    cap = cv2.VideoCapture(0)
    have_color = -1
    voxel_size = 0.3
    # count = 0
    while True:
        _, frame = cap.read()

        img_cropped = frame[centry_x - delete_r:centry_x + delete_r, centry_y - delete_r:centry_y + delete_r]
        # cv2.imwrite("test" + str(count) + ".png", img_cropped)
        # count += 1
        depth_object = get_depth(img_cropped)
        if have_color == -1:
            colors = [(1,0.7,0.2) for i in range(len(depth_object))]
            have_color = 0

        points3d.points = o3d.utility.Vector3dVector(depth_object)
        points3d.colors = o3d.utility.Vector3dVector(colors)

        points3d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=30))

        vis.update_geometry(points3d)

        if to_reset:
            vis.reset_view_point(True)
            to_reset = False

        vis.poll_events()
        vis.update_renderer()

if __name__ == "__main__":
    main()
