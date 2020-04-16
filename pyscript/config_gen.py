import numpy as np
from PIL import Image
import json
import os

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
    dst_dir, dst_img_info, 
    control_point_dim,
    src_pts, dst_pts,
    dst_cfg_fname):

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

