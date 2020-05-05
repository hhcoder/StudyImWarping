import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as pltimage
import os

def plt_show(fig):
    plt.ion()
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=False)

def show(dst_dir, master_location_fname, dbg_location_fname):
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

    plt_show(fig)
    fig.save(os.path.join(dst_dir, dbg_location_fname + "_result_fig.png"))

    im_src_fpath = os.path.join(
            debug_setting["dir"], 
            debug_setting["dumped_png_images"]["src_img_fname"]+debug_setting["dumped_png_images"]["fext"])

    pltimage.imsave(im_src_fpath, im_src, cmap='gray')

    im_dst_fpath = os.path.join(
            debug_setting["dir"], 
            debug_setting["dumped_png_images"]["dst_img_fname"]+debug_setting["dumped_png_images"]["fext"])

    pltimage.imsave(im_dst_fpath, im_dst, cmap='gray')

    return fig, axes

