import numpy as np
import json
import os
import matplotlib.pyplot as plt

def read(dst_dir, master_location_fname):
    np_format_dict = {'raw8': np.uint8,
                    'float': np.float}

    master_file_location = json.load(open(os.path.join(dst_dir, master_location_fname+".json")))

    input_data = json.load(open(os.path.join(master_file_location["dir"], master_file_location["input_data"])))
    driver_setting = json.load(open(os.path.join(master_file_location["dir"], master_file_location["driver_setting"])))
    debug_setting = json.load(open(os.path.join(master_file_location["dir"], master_file_location["debug_setting"])))

    im_src = np.fromfile(
        os.path.join(input_data["src"]["image_bin"]["dir"], 
                     input_data["src"]["image_bin"]["name"]+input_data["src"]["image_bin"]["bin_ext"]), 
        np_format_dict[driver_setting["src"]["image_format"]["format"]]).astype(np.intc)

    im_src = np.reshape(
        im_src, 
        (driver_setting["src"]["image_format"]["height"], 
         driver_setting["src"]["image_format"]["width"]))

    im_dst = np.fromfile(
        os.path.join(input_data["dst"]["image_bin"]["dir"], 
                     input_data["dst"]["image_bin"]["name"]+input_data["dst"]["image_bin"]["bin_ext"]), 
        np_format_dict[driver_setting["dst"]["image_format"]["format"]]).astype(np.intc)

    im_dst = np.reshape(
        im_dst, 
        (driver_setting["dst"]["image_format"]["height"], 
         driver_setting["dst"]["image_format"]["width"]))

    ctrl_pts_info = json.load(open(os.path.join(debug_setting["dir"], debug_setting["dumped_control_points"]["fname"]+debug_setting["dumped_control_points"]["fext"])))

    src_x = np.array(ctrl_pts_info["src_x"])
    src_y = np.array(ctrl_pts_info["src_y"])
    dst_x = np.array(ctrl_pts_info["dst_x"])
    dst_y = np.array(ctrl_pts_info["dst_y"])

    return {"src": { "img": im_src, 
                     "x": src_x, 
                     "y": src_y, },
            "dst": { "img": im_dst, 
                      "x": dst_x, 
                      "y": dst_y} }
