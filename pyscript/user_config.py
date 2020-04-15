#Run Image Warping Script (place holder)
import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as pltimage
import math
from util import img_loader

# import matplotlib.image as mpimg

# class img_desc:
#     data_format = {'raw8': np.uint8,
#                     'float': np.float}

#     def __init__(self, cols=None, rows=None):
#         if cols is not None and rows is not None:
#             self.im = np.empty((cols, rows), dtype=np.intc)
#             self.format_desc = {
#                 "width": self.im.shape[1], 
#                 "height": self.im.shape[0], 
#                 "format": "raw8" }

#     def read_encoded(self, dir, fname, fext):
#         fpath = os.path.join(dir, fname + fext)
#         self.im = np.array(Image.open(fpath).convert("L"), dtype=np.intc)
#         self.format_desc = {
#             "width": self.im.shape[1], 
#             "height": self.im.shape[0], 
#             "format": "raw8" }
    
#     def rows(self):
#         return self.im.shape[0]

#     def cols(self):
#         return self.im.shape[1]

#     def read_binary(self, dir, fname, bin_ext, json_ext = ".json"):
#         self.format = json.load(open(os.path.join(dir, fname+json_ext))) 
#         np_fmt = img_desc.data_format[self.format["format"]]
#         b = np.fromfile(os.path.join(dir, fname+bin_ext), np_fmt) 
#         self.im = np.reshape(b.astype(np.intc), (self.format["height"], self.format["width"]))

#     def fwrite(self, dir, fname, bin_ext, json_ext = ".json"): 
#         if not os.path.exists(dir):
#             os.makedirs(dir)
#         with open(os.path.join(dir, fname+bin_ext), "wb") as f:
#             self.im.astype(np.uint8).tofile(f)
#         with open(os.path.join(dir, fname+json_ext), 'w') as f:
#             self.file_desc = {
#                 "folder": dir,
#                 "name": fname,
#                 "bin_ext": bin_ext,
#                 "json_ext": ".json",
#                 "width": self.cols(),
#                 "height": self.rows(),
#                 "format": "raw8"}
#             json.dump(self.file_desc, f, indent=4)

#     def show(self, axes):
#         axes.imshow(self.im, cmap="gray")

# class ctrl_pts_desc:
#     def __init__(self, cols, rows):
#         self.x = np.empty((rows,cols),dtype=np.intc)
#         self.y = np.empty((rows,cols),dtype=np.intc)
    
#     def rows(self):
#         return self.x.shape[0]
    
#     def cols(self):
#         return self.x.shape[1]

#     @staticmethod
#     def distance_to_center(xloc, yloc, width_half, height_half):
#         idx_dist = math.sqrt((xloc-width_half)**2 + (yloc-height_half)**2)
#         diag_dist = math.sqrt(width_half**2 + height_half**2)
#         dist =  idx_dist/diag_dist 
#         return math.sqrt(math.cos( dist * math.pi / 4))

#     def setup_fish_eye_ctrl_pts(self, img_width, img_height):
#         yidx = np.linspace(0, img_height, num=self.rows()).astype(int)
#         xidx = np.linspace(0, img_width, num=self.cols()).astype(int)
#         n = 0
#         for j in range(0, self.rows()):
#             for i in range(0, self.cols()):
#                 factor = ctrl_pts_desc.distance_to_center(
#                     xidx[i], yidx[j], img_width/2, img_height/2)
#                 self.x[j][i] = (xidx[i] - img_width/2) * factor + img_width/2
#                 self.y[j][i] = (yidx[j] - img_height/2)* factor + img_height/2
#                 n = n+1
    
#     def setup_grid_ctrl_pts(self, img_width, img_height):
#         yidx = np.linspace(0, img_height, num=self.rows()).astype(int)
#         xidx = np.linspace(0, img_width, num=self.cols()).astype(int)
#         n = 0
#         for j in range(0, self.rows()):
#             for i in range(0, self.cols()):
#                 self.x[j][i] = xidx[i]
#                 self.y[j][i] = yidx[j]
#                 n = n+1
    
#     def show(self, axes):
#         for j in range(0, self.rows()):
#             for i in range(0, self.cols()):
#                 axes.plot(self.x[j][i], self.y[j][i], "oy")

# with open(os.path.join(dst_dir, dst_fname+".json"), 'w') as f:
#     file_desc = {
#         "folder": dst_dir,
#         "name": dst_fname,
#         "bin_ext": ".bin",
#         "json_ext": ".json",
#         "width": cols, 
#         "height": rows, 
#         "stride": stride,
#         "format": dst_format }
#     json.dump(file_desc, f, indent=4)


def distance_to_center(xloc, yloc, width_half, height_half):
    idx_dist = math.sqrt((xloc-width_half)**2 + (yloc-height_half)**2)
    diag_dist = math.sqrt(width_half**2 + height_half**2)
    dist =  idx_dist/diag_dist 
    return math.sqrt(math.cos( dist * math.pi / 3.6))

def ctrl_pts_fish_eye(img_width, img_height, ctrl_cols, ctrl_rows):
    xidx = np.linspace(0, img_width-1, num=ctrl_cols).astype(int)
    yidx = np.linspace(0, img_height-1, num=ctrl_rows).astype(int)
    x = np.empty((ctrl_rows, ctrl_cols),dtype=np.intc)
    y = np.empty((ctrl_rows, ctrl_cols),dtype=np.intc)
    n = 0
    for j in range(0, ctrl_rows):
        for i in range(0, ctrl_cols):
            factor = distance_to_center(xidx[i], yidx[j], img_width/2, img_height/2)
            x[j][i] = (xidx[i] - img_width/2) * factor + img_width/2
            y[j][i] = (yidx[j] - img_height/2)* factor + img_height/2
            n = n+1
    return {"x": x, "y": y}

def ctrl_pts_rect_grid(img_width, img_height, ctrl_cols, ctrl_rows):
    xidx = np.linspace(0, img_width-1, num=ctrl_cols).astype(int)
    yidx = np.linspace(0, img_height-1, num=ctrl_rows).astype(int)
    x = np.empty((ctrl_rows, ctrl_cols),dtype=np.intc)
    y = np.empty((ctrl_rows, ctrl_cols),dtype=np.intc)
    n = 0
    for j in range(0, ctrl_rows):
        for i in range(0, ctrl_cols):
            x[j][i] = xidx[i]
            y[j][i] = yidx[j]
            n = n+1
    return {"x": x, "y": y}

def decode_image_to_binary_format(
    src_dir, src_fname, src_fext, 
    dst_dir, dst_fname, dst_format, dst_bin_ext=".bin"):
    fpath = os.path.join(src_dir, src_fname + src_fext)
    im = np.array(Image.open(fpath).convert("L"), dtype=np.intc)
    (cols, rows, stride) = (im.shape[1], im.shape[0], im.shape[1])
    with open(os.path.join(dst_dir, dst_fname+dst_bin_ext), "wb") as f:
        im.astype(np.uint8).tofile(f)
    # with open(os.path.join(dst_dir, dst_fname+dst_json_ext), "w") as f:
    #     desc = { "width": cols, 
    #              "height": rows, 
    #              "stride": stride, 
    #              "format": "raw8" }
    #     json.dump(desc, f, indent=4)
    return {"width": cols, "height": rows, "stride": stride, "format": dst_format}

def generate_user_config(
    src_img_info,
    dst_dir,dst_img_info, control_point_dim,
    dst_cfg_fname):

    src_pts = ctrl_pts_fish_eye(
        src_img_info["width"], src_img_info["height"], 
        control_point_dim[0], control_point_dim[1])

    dst_pts = ctrl_pts_rect_grid(
        dst_img_info["width"], dst_img_info["height"], 
        control_point_dim[0], control_point_dim[1])

    out_dict = {
        "src":
        {
            "image_format": src_img_info,
            "control_points":
            {
                "cols": control_point_dim[0],
                "rows": control_point_dim[1],
                "x": src_pts["x"].tolist(),
                "y": src_pts["y"].tolist()
            },
        },
        "dst":
        {
            "image_format": dst_img_info,
            "control_points":
            {
                "cols": control_point_dim[0],
                "rows": control_point_dim[1],
                "x": dst_pts["x"].tolist(),
                "y": dst_pts["y"].tolist()
            },
        }
    }

    with open(os.path.join(dst_dir, dst_cfg_fname+".json"), 'w') as f:
        json.dump(out_dict, f, indent=4)

def generate_input_data(
    src_dir, src_img_fname,
    dst_dir, dst_img_fname,
    input_data_fname):
    out_dict = {
        "src":
        {
            "image_bin":
            {
                "dir": src_dir,
                "name": src_img_fname,
                "bin_ext": ".bin",
            }
        },
        "dst":
        {
            "image_bin":
            {
                "dir": dst_dir,
                "name": dst_img_fname,
                "bin_ext": ".bin",
            }
        }
    }
    with open(os.path.join(dst_dir, input_data_fname+".json"), 'w') as f:
        json.dump(out_dict, f, indent=4)

def generate_driver_setting(
    src_dir, src_cfg_fname, 
    dst_dir, dst_setting_fname):
    cfg_info = json.load(open(os.path.join(src_dir, src_cfg_fname+".json")))
    with open(os.path.join(dst_dir, dst_setting_fname+".json"), 'w') as f:
        json.dump(cfg_info, f, indent=4)


def generate_dbg_setting(dst_dir, dbg_setting_fname):
    dbg_dict = {
        "dir": dst_dir,
        "dumped_control_points":
        {
            "fname": dbg_setting_fname+"_ctrl_pts",
            "fext": ".json"
        },
        "dumped_png_images":
        {
            "src_img_fname": dbg_setting_fname+"_src_img",
            "dst_img_fname": dbg_setting_fname+"_dst_img",
            "fext": ".png",
        }
    }

    with open(os.path.join(dst_dir, dbg_setting_fname+".json"), 'w') as f:
        json.dump(dbg_dict, f, indent=4)

def generate_master_setting( 
    src_dir, 
    input_data_fname, 
    user_config_fname, 
    driver_setting_fname, 
    dbg_setting_fname,
    dst_dir, master_location_fname):
    loc_info = {
        "dir": src_dir,
        "input_data": input_data_fname + ".json",
        "user_config": user_config_fname + ".json",
        "driver_setting": driver_setting_fname + ".json",
        "debug_setting": dbg_setting_fname + ".json"
    }
    with open(os.path.join(dst_dir, master_location_fname+".json"), 'w') as f:
        json.dump(loc_info, f, indent=4)

def plt_show_and_save(fig, fig_fpath):
    plt.ion()
    plt.draw()
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show(block=False)
    plt.pause(0.1)
    fig.savefig(fig_fpath)
    plt.close("all")

def show_warpping_result(dst_dir, master_location_fname, dbg_location_fname):
    np_format_dict = {'raw8': np.uint8,
                    'float': np.float}

    file_location = json.load(open(os.path.join(dst_dir, master_location_fname+".json")))

    input_data = json.load(open(os.path.join(file_location["dir"], file_location["input_data"])))
    driver_setting = json.load(open(os.path.join(file_location["dir"], file_location["driver_setting"])))
    debug_setting = json.load(open(os.path.join(file_location["dir"], file_location["debug_setting"])))

    im_src = np.fromfile(
        os.path.join(input_data["src"]["image_bin"]["dir"], input_data["src"]["image_bin"]["name"]+input_data["src"]["image_bin"]["bin_ext"]), 
        np_format_dict[driver_setting["src"]["image_format"]["format"]]).astype(np.intc)

    im_src = np.reshape(
        im_src, 
        (driver_setting["src"]["image_format"]["height"], 
         driver_setting["src"]["image_format"]["width"]))

    im_dst = np.fromfile(
        os.path.join(input_data["dst"]["image_bin"]["dir"], input_data["dst"]["image_bin"]["name"]+input_data["dst"]["image_bin"]["bin_ext"]), 
        np_format_dict[driver_setting["dst"]["image_format"]["format"]]).astype(np.intc)

    im_dst = np.reshape(
        im_dst, 
        (driver_setting["dst"]["image_format"]["height"], 
         driver_setting["dst"]["image_format"]["width"]))

    fig,axes = plt.subplots(1,2)
    axes[0].imshow(im_src, cmap="gray")
    axes[1].imshow(im_dst, cmap="gray")

    ctrl_pts_info = json.load(open(os.path.join(debug_setting["dir"], debug_setting["dumped_control_points"]["fname"]+debug_setting["dumped_control_points"]["fext"])))
    src_x = np.array(ctrl_pts_info["src_x"])
    src_y = np.array(ctrl_pts_info["src_y"])
    for j in range(0, src_x.shape[0]):
        for i in range(0, src_x.shape[1]):
            axes[0].plot(src_x[j][i], src_y[j][i], "oy")

    dst_x = np.array(ctrl_pts_info["dst_x"])
    dst_y = np.array(ctrl_pts_info["dst_y"])
    for j in range(0, dst_x.shape[0]):
        for i in range(0, dst_x.shape[1]):
            axes[1].plot(dst_x[j][i], dst_y[j][i], "ob")

    plt_show_and_save(fig, os.path.join(dst_dir, dbg_location_fname + "_result_fig.png"))

    im_src_fpath = os.path.join(
            debug_setting["dir"], 
            debug_setting["dumped_png_images"]["src_img_fname"]+debug_setting["dumped_png_images"]["fext"])

    pltimage.imsave(im_src_fpath, im_src, cmap='gray')

    im_dst_fpath = os.path.join(
            debug_setting["dir"], 
            debug_setting["dumped_png_images"]["dst_img_fname"]+debug_setting["dumped_png_images"]["fext"])

    pltimage.imsave(im_dst_fpath, im_dst, cmap='gray')

def warp_process_command_line(master_setting_file_path):
    SolutionDir = "C:/HHWork/ImWarping/Development/"
    BuildDir = "build/"
    Platform = "Win32"
    Configuration = "Debug"
    ProjectName = "core_lib"
    ExecutableExt = ".exe"
    execute_path = SolutionDir + BuildDir + Platform + "/" + Configuration + "/" + ProjectName + ExecutableExt
    command_line = execute_path + " " + master_setting_file_path
    os.system(command_line)


#PyDefault
# src_dir = "C:/HHWork/ImWarping/Data/Input/PyDefault/" 
# src_png_fname = "NIR_2020-03-18-05-08-04-429"
# src_png_ext = ".png"
# dst_dir = "C:/HHWork/ImWarping/Data/Output/PyDefault/"

# src_dir = "C:/HHWork/ImWarping/Data/Input/Internet/" 
# src_png_fname = "girl-with-balloon"
# src_png_ext = ".png"
# dst_dir = "C:/HHWork/ImWarping/Data/Output/Internet/"

src_dir = "C:/HHWork/ImWarping/Data/Input/Grid/" 
src_png_fname = "grid"
src_png_ext = ".png"
dst_dir = "C:/HHWork/ImWarping/Data/Output/Grid/"

src_img_fname = src_png_fname + "_src_img"
src_img_format = "raw8"

dst_img_fname = src_png_fname + "_warpped_img"
dst_img_format = "raw8"

control_point_dim = (7, 9)

# generate_all_config(src_dir, src_png_fname, src_png_ex)

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

user_config_fname = src_png_fname + "_user_config"
input_data_fname = src_png_fname + "_input_data"
driver_setting_fname = src_png_fname + "_driver_setting"
master_location_fname = src_png_fname + "_main"
dbg_setting_fname = src_png_fname + "_debug_setting"

src_img_info = decode_image_to_binary_format(
    src_dir, src_png_fname, src_png_ext, 
    dst_dir, src_img_fname, src_img_format)

# No scaling or color conversion for now
dst_img_info = src_img_info

generate_user_config(
    src_img_info,
    dst_dir, dst_img_info, control_point_dim,
    user_config_fname)

generate_input_data(
    dst_dir, src_img_fname,
    dst_dir, dst_img_fname,
    input_data_fname)

generate_driver_setting(
    dst_dir, user_config_fname, 
    dst_dir, driver_setting_fname)

generate_dbg_setting(
    dst_dir, 
    dbg_setting_fname)

generate_master_setting(
    dst_dir, 
    input_data_fname, 
    user_config_fname, 
    driver_setting_fname, 
    dbg_setting_fname,
    dst_dir, master_location_fname)

warp_process_command_line(os.path.join(dst_dir, master_location_fname+".json"))

show_warpping_result(dst_dir, master_location_fname, dbg_setting_fname)
