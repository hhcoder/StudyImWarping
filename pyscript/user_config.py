#Run Image Warping Script (place holder)
import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    return math.sqrt(math.cos( dist * math.pi / 4))

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
    dst_dir, dst_fname, dst_format, dst_bin_ext=".bin", dst_json_ext=".json"):
    fpath = os.path.join(src_dir, src_fname + src_fext)
    im = np.array(Image.open(fpath).convert("L"), dtype=np.intc)
    (cols, rows, stride) = (im.shape[1], im.shape[0], im.shape[1])
    with open(os.path.join(dst_dir, dst_fname+dst_bin_ext), "wb") as f:
        im.astype(np.uint8).tofile(f)
    with open(os.path.join(dst_dir, dst_fname+dst_json_ext), "w") as f:
        desc = { "width": cols, 
                 "height": rows, 
                 "stride": stride, 
                 "format": "raw8" }
        json.dump(desc, f, indent=4)
    return (cols, rows, stride)

def generate_user_config(
    src_img_dim, src_img_format,
    dst_dir, dst_img_dim, dst_img_format, control_point_dim,
    dst_cfg_fname):

    src_pts = ctrl_pts_fish_eye(src_img_dim[0], src_img_dim[1], control_point_dim[0], control_point_dim[1])

    dst_pts = ctrl_pts_rect_grid(src_img_dim[0], src_img_dim[1], control_point_dim[0], control_point_dim[1])

    out_dict = {
        "src":
        {
            "image_format": 
            {
                "width": src_img_dim[0],
                "height": src_img_dim[1],
                "stride": src_img_dim[2],
                "format": src_img_format, 
            },
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
            "image_format": 
            {
                "width": dst_img_dim[0],
                "height": dst_img_dim[1],
                "stride": dst_img_dim[2],
                "format": dst_img_format
            },
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
    dst_input_fname):
    out_dict = {
        "src":
        {
            "image_bin":
            {
                "dir": src_dir,
                "name": src_img_fname,
                "bin_ext": ".bin",
                "json_ext": ".json"
            }
        },
        "dst":
        {
            "image_bin":
            {
                "dir": dst_dir,
                "name": dst_img_fname,
                "bin_ext": ".bin",
                "json_ext": ".json"
            }
        }
    }
    with open(os.path.join(dst_dir, dst_input_fname+".json"), 'w') as f:
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
        "fname": dbg_setting_fname,
        "fext": ".json" }

    with open(os.path.join(dst_dir, dbg_setting_fname+".json"), 'w') as f:
        json.dump(dbg_dict, f, indent=4)

def generate_input_file_location( 
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
        "dbg_setting": dbg_setting_fname + ".json"
    }
    with open(os.path.join(dst_dir, master_location_fname+".json"), 'w') as f:
        json.dump(loc_info, f, indent=4)

def show_warpping_result(dst_dir, master_location_fname):
    file_location = json.load(open(os.path.join(dst_dir, master_location_fname+".json")))

    input_data = json.load(open(os.path.join(file_location["dir"], file_location["input_data"])))
    driver_setting = json.load(open(os.path.join(file_location["dir"], file_location["driver_setting"])))

    img_src = img_loader()
    img_src.read_binary(input_data["src"]["image_bin"]["dir"], input_data["src"]["image_bin"]["name"], ".bin", ".json")
    img_dst = img_loader()
    img_dst.read_binary(input_data["dst"]["image_bin"]["dir"], input_data["dst"]["image_bin"]["name"], ".bin", ".json")
    fig,axes = plt.subplots(1,2)
    img_src.show(axes[0])
    img_dst.show(axes[1])
    # debug_setting = json.load(open(os.path.join(file_location["dir"], file_location["debug_setting"])))
    # debug_result = json.load(open(os.path.join(debug_setting["dir"], debug_setting["fname"]+debug_setting["fext"])))
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.show(block=True)
    plt.pause(0.001)


#PyDefault
# src_dir = "C:/HHWork/ImWarping/Data/Input/PyDefault/" 
# src_png_fname = "NIR_2020-03-18-05-08-04-429"
# src_png_ext = ".png"
# dst_dir = "C:/HHWork/ImWarping/Data/Output/PyDefault/"

src_dir = "C:/HHWork/ImWarping/Data/Input/Internet/" 
src_png_fname = "girl-with-balloon"
src_png_ext = ".png"
dst_dir = "C:/HHWork/ImWarping/Data/Output/Internet/"

src_img_fname = src_png_fname + "_src_img"
src_img_format = "gray8"

dst_img_fname = src_png_fname + "_warpped_img"
dst_img_format = "gray8"

control_point_dim = (4, 5)

# generate_all_config(src_dir, src_png_fname, src_png_ex)

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

user_config_fname = src_png_fname + "_user_config"
input_data_fname = src_png_fname + "_input_data"
driver_setting_fname = src_png_fname + "_driver_setting"
master_location_fname = src_png_fname + "_main"
dbg_setting_fname = src_png_fname + "_dbg_setting"

src_dim = decode_image_to_binary_format(
    src_dir, src_png_fname, src_png_ext, 
    dst_dir, src_img_fname, src_img_format)

# No scaling for now
dst_dim = src_dim

generate_user_config(
    src_dim, src_img_format,
    dst_dir, dst_dim, dst_img_format, control_point_dim,
    user_config_fname)

generate_input_data(
    dst_dir, src_img_fname,
    dst_dir, dst_img_fname,
    input_data_fname)

generate_driver_setting(
    dst_dir, user_config_fname, 
    dst_dir, driver_setting_fname)

generate_dbg_setting(dst_dir, dbg_setting_fname)

generate_input_file_location(
    dst_dir, input_data_fname, user_config_fname, driver_setting_fname, dbg_setting_fname,
    dst_dir, master_location_fname)

show_warpping_result(dst_dir, master_location_fname)

# fig,axes = plt.subplots(1,2)
# img_src.fwrite(dst_dir, src_fname, ".bin")
# src_tiles = tiles_desc(3, 4)
# src_tiles.setup_with_ctrl_pts(src_pts)

# img_src.show(axes[0])
# src_pts.show(axes[0])

# img_dst.show(axes[1])

# img_dst.show(axes[1])
# dst_pts.show(axes[1])

# plt.ion()
# plt.draw()
# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
# plt.show(block=True)
# plt.pause(0.001)
# class tiles_desc:
#     def __init__(self, cols, rows):
#         self.t = np.empty((rows, cols, 2, 4), dtype=np.intc)
    
#     def rows(self):
#         return self.t.shape[0]
#     def cols(self):
#         return self.t.shape[1]

#     def setup_with_ctrl_pts(self, ctrl_pts):
#         for j in range(self.rows()):
#             for i in range (self.cols()):
#                 xmax = max( (ctrl_pts.x[j][i], ctrl_pts.x[j][i+1], ctrl_pts.x[j+1][i], ctrl_pts.x[j+1][i+1]) )
#                 xmin = min( (ctrl_pts.x[j][i], ctrl_pts.x[j][i+1], ctrl_pts.x[j+1][i], ctrl_pts.x[j+1][i+1]) )
#                 ymax = max( (ctrl_pts.y[j][i], ctrl_pts.y[j][i+1], ctrl_pts.y[j+1][i], ctrl_pts.y[j+1][i+1]) )
#                 ymin = min( (ctrl_pts.y[j][i], ctrl_pts.y[j][i+1], ctrl_pts.y[j+1][i], ctrl_pts.y[j+1][i+1]) )
#                 self.t[j, i, 0, :] = [xmin, xmax, ymin, ymax]

