import os
import torch.utils.data as Data
from torch.utils.data import Dataset

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# define MLP module
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

        # dropout
        self.drop_out = nn.Dropout(0.02)

    def forward(self, x):
        x = x.view(-1, 5)
        x = self.drop_out(torch.tanh(self.fc1(x)))
        x = self.drop_out(torch.tanh(self.fc2(x)))
        x = self.drop_out(torch.tanh(self.fc3(x)))
        x = self.drop_out(torch.tanh(self.fc4(x)))

        x = self.fc5(x)
        return x


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("new folder")
    else:
        print("There is this folder!")


# Training data set path # not split
train_data_set_path = "/home/ljy/tactile_reconstruction/code/2generate_train_test_dataset/train_circles_vpaper.xls"

# Load the data, convert it into a numpy array, and then convert it into a Dataset
train_data_pd = pd.read_excel(train_data_set_path, sheet_name=None)
keys = list(train_data_pd.keys())

number = 0
data_concat = pd.DataFrame()
for key in keys:
    print(number)
    number += 1
    data_this_key = train_data_pd[key]
    data_concat = pd.concat([data_concat, data_this_key])

train_data_np = data_concat.to_numpy(dtype=np.float32)

data_x = train_data_np[:, :5]
data_y = train_data_np[:, -2:]

data_x[:, :2] = data_x[:, :2] / 320
data_x[:, 2:5] = data_x[:, 2:5] / 255

# Test data set address # not split
test_data_set_path = "/home/ljy/tactile_reconstruction/code/2generate_train_test_dataset/test_circles_vpaper.xls"

# Load the data, convert it into a numpy array, and then convert it into a Dataset
test_data_pd = pd.read_excel(test_data_set_path, sheet_name=None)
test_keys = list(test_data_pd.keys())

test_number = 0
test_data_concat = pd.DataFrame()
for key in test_keys:
    print(test_number)
    test_number += 1
    data_this_key = test_data_pd[key]
    test_data_concat = pd.concat([test_data_concat, data_this_key])

test_data_np = test_data_concat.to_numpy(dtype=np.float32)

test_data_x = test_data_np[:, :5]
test_data_y = test_data_np[:, -2:]

test_data_x[:, :2] = test_data_x[:, :2] / 320
test_data_x[:, 2:5] = test_data_x[:, 2:5] / 255

X_train, X_test, y_train, y_test = data_x, test_data_x, data_y, test_data_y

train_xt = torch.from_numpy(X_train)
train_yt = torch.from_numpy(y_train)

test_xt = torch.from_numpy(X_test)
test_yt = torch.from_numpy(y_test)

train_data = Data.TensorDataset(train_xt, train_yt)
test_data = Data.TensorDataset(test_xt, test_yt)

# loop 1
learning_rate = 0.00224
file_name = 120500 # 2022 12 05
n_epoch = 500
save_epoch = 10
batch_size = 4000

train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=1)
model = MyMLP().cuda()
# print(model)
# Define loss function and optimizer
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train
train_loss_all = []
test_loss_all = []
for epoch in range(n_epoch):
    train_loss = 0.0
    train_num = 0
    for (b_x, b_y) in train_loader:
        b_x, b_y = b_x.cuda(), b_y.cuda()
        output = model(b_x)
        loss = criterion(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().item() * b_x.size(0)
        train_num += b_x.size(0)
    train_loss_all.append(train_loss / train_num)
    print("Epoch:{} \t Training Loss: {:.6f}".format(epoch+1, train_loss_all[-1]))

    if (epoch+1) % save_epoch == 0 or epoch == 0:
        # file name
        file = "/home/ljy/tactile_reconstruction/code/3train_model/train_log_new_dataset/" + str(file_name)
        mkdir(file)     # use the function

        # save EXCEL
        dict_data = {}
        dict_data["epoch"] = list(range(len(train_loss_all)))
        dict_data["error"] = train_loss_all

        df = pd.DataFrame(dict_data)

        with pd.ExcelWriter("/home/ljy/tactile_reconstruction/code/3train_model/train_log_new_dataset/train_log_120500.xlsx", mode='a') as writer:
            df.to_excel(writer, sheet_name=str(file_name), index=False)
            writer.save()

        # save the module
        torch.save(model.state_dict(), file + "/MLP6.pkl")

        # test
        model_test = MyMLP().cuda()
        model_test.load_state_dict(torch.load(file + "/MLP6.pkl"))

        model_test.eval()

        pre_y = model_test(test_xt.cuda())
        pre_y = pre_y.cpu().data.numpy()
        mae = mean_absolute_error(y_test, pre_y)
        mse = mean_squared_error(y_test, pre_y)
        test_loss_all.append(mse)
        print("the error in test_set is : MAE = ", mae)
        print("the error in test_set is : MSE = ", mse)

        #  Draw a line chart
        plt.plot(list(range(len(train_loss_all))), train_loss_all)
        plt.savefig(file + "/train_loss_6.jpg")
        plt.clf()

        plt.plot(list(range(save_epoch, len(test_loss_all)*save_epoch+1, save_epoch)), test_loss_all)
        plt.savefig(file + "/test_loss_6.jpg")
        plt.clf()
        # plt.show()

        # Record training parameters
        f = open(file + "/log.txt", mode='w')
        f.write(f"mse = {mse} + mae = {mae} + batch_size = {batch_size} + learning_rate = {learning_rate} + n_epoch = {epoch}")
        f.close()

        # save EXCEL
        dict_data = {}
        dict_data["epoch"] = list(range(len(test_loss_all)))
        dict_data["error"] = test_loss_all

        df = pd.DataFrame(dict_data)

        with pd.ExcelWriter("/home/ljy/tactile_reconstruction/code/3train_model/train_log_new_dataset/test_log_120500.xlsx", mode='a') as writer:
            df.to_excel(writer, sheet_name=str(file_name), index=False)
            writer.save()

        file_name += 1

# clear gpu cache
torch.cuda.empty_cache()




