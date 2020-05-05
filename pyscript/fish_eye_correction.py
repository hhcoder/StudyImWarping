#Run Image Warping Script (place holder)
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as pltimage
import math

import config_gen as cfg
import cmdline
import result_disp

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
    for j in range(0, ctrl_rows):
        for i in range(0, ctrl_cols):
            x[j][i] = xidx[i]
            y[j][i] = yidx[j]
    return {"x": x, "y": y}


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

src_img_info = cfg.decode_image_to_binary_format(
    src_dir, src_png_fname, src_png_ext, 
    dst_dir, src_img_fname, src_img_format)

# No scaling or color conversion for now
dst_img_info = src_img_info

src_pts = ctrl_pts_fish_eye(
    src_img_info["width"], src_img_info["height"], 
    control_point_dim[0], control_point_dim[1])

dst_pts = ctrl_pts_rect_grid(
    dst_img_info["width"], dst_img_info["height"], 
    control_point_dim[0], control_point_dim[1])

cfg.generate_user_config(
    src_img_info,
    dst_dir, dst_img_info, 
    control_point_dim,
    src_pts,
    dst_pts,
    user_config_fname)

cfg.generate_input_data(
    dst_dir, src_img_fname,
    dst_dir, dst_img_fname,
    input_data_fname)

cfg.generate_driver_setting(
    dst_dir, user_config_fname, 
    dst_dir, driver_setting_fname)

cfg.generate_dbg_setting(
    dst_dir, 
    dbg_setting_fname)

cfg.generate_master_setting(
    dst_dir, 
    input_data_fname, 
    user_config_fname, 
    driver_setting_fname, 
    dbg_setting_fname,
    dst_dir, master_location_fname)

cmdline.exe(os.path.join(dst_dir, master_location_fname+".json"))

result_disp.show(dst_dir, master_location_fname, dbg_setting_fname)
