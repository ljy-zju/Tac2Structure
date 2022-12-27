from tkinter import *
from PIL import Image,ImageTk,ImageDraw

import pandas as pd
import time

parameters = {}


def save_csv():
    save_dict = {}
    save_dict["number"] = []
    save_dict["r_x_y"] = []
    for key,value in parameters.items():
        save_dict["number"].append(key)
        save_dict["r_x_y"].append(value)

    data_df = pd.DataFrame(save_dict)
    data_df.to_excel('data_df.xls', index=False)
    print("Save csv ok!")


def copy_param():
    global number
    global parameters
    r = int(v.get())
    x = int(v2.get())
    y = int(v3.get())
    root.clipboard_clear()
    root.clipboard_append(f"{r}-{y}-{x}")

    parameters[f"{number}"] = f"{r}-{y}-{x}"
    print(f"write param of pic {number}")


def change_pic():
    global number
    number = number + 1
    picture_name_vis.set(f"{number:>05}"  +  ".png")
    draw_circles()


def change_pic_back():
    global number
    number = number - 1
    picture_name_vis.set(f"{number:>05}"  +  ".png")
    draw_circles()


def draw_circles(*args):
    time.sleep(0.01)
    r = int(v.get())
    x = int(v2.get())
    y = int(v3.get())

    canshu_vis.set(f"r,x,y : {r},{y},{x}")

    load = Image.open("/home/ljy/tactile_reconstruction/code/pic_data_set/for_test/" + f"{number:>05}"  +  ".png")

    draw = ImageDraw.Draw(load)
    draw.ellipse((x-r,y-r,x+r,y+r))

    render = ImageTk.PhotoImage(load)

    img = Label(root, image=render)
    img.image = render
    img.place(x=0,y=0)

if __name__ == '__main__':
    number = 0
    root = Tk()
    root.geometry("900x500")
    root.title("Tool Box")

    r_vis = StringVar()
    r_label = Label(root,anchor=E,textvariable=r_vis,state="active")
    r_vis.set("R :")
    r_label.place(x=720,y=16)

    v = DoubleVar(name="r")
    scale = Scale(root, variable = v, from_ = 1, to = 52, orient = HORIZONTAL,length=150,command=draw_circles)
    scale.pack(anchor='e')

    y_vis = StringVar()
    y_label = Label(root,anchor=E,textvariable=y_vis,state="active")
    y_vis.set("Y :")
    y_label.place(x=720,y=52)

    v2 = DoubleVar(name="y")
    scale = Scale(root, variable = v2, from_ = 1, to = 640, orient = HORIZONTAL,length=150,command=draw_circles)
    scale.pack(anchor='e')

    x_vis = StringVar()
    x_label = Label(root,anchor=E,textvariable=x_vis,state="active")
    x_vis.set("X :")
    x_label.place(x=720,y=90)

    v3 = DoubleVar(name="x")
    scale = Scale(root, variable = v3, from_ = 1, to = 480, orient = HORIZONTAL,length=150,command=draw_circles)
    scale.pack(anchor='e')

    button = Button(root, text="change picture", command=change_pic)
    button.place(x=720, y=300) # 900 - 640 = 260 ; 80 + 640 = 720

    button_change_back = Button(root, text="change back   ", command=change_pic_back)
    button_change_back.place(x=720, y=400)

    button_copy = Button(root, text="copy param    ", command=copy_param)
    button_copy.place(x=720, y=350)

    button_save_csv = Button(root, text="save csv    ", command=save_csv)

    button_save_csv.place(x=720, y=450)

    canshu_vis = StringVar()
    canshu_label = Label(root,anchor=E,textvariable=canshu_vis,state="active")
    canshu_vis.set("r,x,y : 0,0,0")
    canshu_label.place(x=720,y=200)

    picture_name_vis = StringVar()
    picture_name_label = Label(root, anchor=E, textvariable=picture_name_vis, state="active")
    picture_name_vis.set(f"{number:>05}"  +  ".png")
    picture_name_label.place(x=720, y=250)



    draw_circles()
    root.mainloop()
