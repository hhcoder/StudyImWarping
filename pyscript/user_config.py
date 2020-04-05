#Run Image Warping Script (place holder)
import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
# import matplotlib.image as mpimg

class img_desc:
    data_format = {'raw8': np.uint8,
                    'float': np.float}

    def __init__(self, cols=0, rows=0):
        self.im = np.array((cols, rows), np.intc)
        self.format_desc = {
            "width": self.im.shape[1], 
            "height": self.im.shape[0], 
            "format": "raw8" }

    def read_encoded(self, dir, fname, fext):
        fpath = os.path.join(dir, fname + fext)
        self.im = np.array(Image.open(fpath).convert("L"), np.intc)
        self.format_desc = {
            "width": self.im.shape[1], 
            "height": self.im.shape[0], 
            "format": "raw8" }
    
    def rows(self):
        return self.im.shape[0]

    def cols(self):
        return self.im.shape[1]

    def read_binary(self, dir, fname, bin_ext, json_ext = ".json"):
        self.format = json.load(open(os.path.join(dir, fname+json_ext))) 
        np_fmt = img_desc.data_format[self.format["format"]]
        b = np.fromfile(os.path.join(dir, fname+bin_ext), np_fmt) 
        self.im = np.reshape(b.astype(np.intc), (self.format["height"], self.format["width"]))

    def fwrite(self, dir, fname, bin_ext, json_ext = ".json"): 
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(os.path.join(dir, fname+bin_ext), "wb") as f:
            self.im.astype(np.uint8).tofile(f)
        with open(os.path.join(dir, fname+json_ext), 'w') as f:
            self.file_desc = {
                "folder": dir,
                "name": fname,
                "bin_ext": bin_ext,
                "json_ext": ".json",
                "width": self.cols(),
                "height": self.rows(),
                "format": "raw8"}
            json.dump(self.file_desc, f, indent=4)

    def show(self, axes):
        axes.imshow(self.im, cmap="gray")

class ctrl_pts_desc:
    def __init__(self, cols, rows):
        self.x = np.ndarray((rows,cols),dtype=np.intc)
        self.y = np.ndarray((rows,cols),dtype=np.intc)
    
    def rows(self):
        return self.x.shape[0]
    
    def cols(self):
        return self.x.shape[1]

    @staticmethod
    def distance_to_center(xloc, yloc, width_half, height_half):
        idx_dist = math.sqrt((xloc-width_half)**2 + (yloc-height_half)**2)
        diag_dist = math.sqrt(width_half**2 + height_half**2)
        dist =  idx_dist/diag_dist 
        return math.sqrt(math.cos( dist * math.pi / 4))

    def setup_fish_eye_ctrl_pts(self, img_width, img_height):
        yidx = np.linspace(0, img_height, num=self.rows()).astype(int)
        xidx = np.linspace(0, img_width, num=self.cols()).astype(int)
        n = 0
        for j in range(0, self.rows()):
            for i in range(0, self.cols()):
                factor = ctrl_pts_desc.distance_to_center(
                    xidx[i], yidx[j], img_width/2, img_height/2)
                self.x[j][i] = (xidx[i] - img_width/2) * factor + img_width/2
                self.y[j][i] = (yidx[j] - img_height/2)* factor + img_height/2
                n = n+1
    
    def setup_grid_ctrl_pts(self, img_width, img_height):
        yidx = np.linspace(0, img_height, num=self.rows()).astype(int)
        xidx = np.linspace(0, img_width, num=self.cols()).astype(int)
        n = 0
        for j in range(0, self.rows()):
            for i in range(0, self.cols()):
                self.x[j][i] = xidx[i]
                self.y[j][i] = yidx[j]
                n = n+1
    
    def show(self, axes):
        for j in range(0, self.rows()):
            for i in range(0, self.cols()):
                axes.plot(self.x[j][i], self.y[j][i], "oy")

class tiles_desc:
    def __init__(self, cols, rows):
        self.t = np.ndarray((rows, cols, 2, 4), dtype=np.intc)
    
    def rows(self):
        return self.t.shape[0]
    def cols(self):
        return self.t.shape[1]

    def setup_with_ctrl_pts(self, ctrl_pts):
        for j in range(self.rows()):
            for i in range (self.cols()):
                xmax = max( (ctrl_pts.x[j][i], ctrl_pts.x[j][i+1], ctrl_pts.x[j+1][i], ctrl_pts.x[j+1][i+1]) )
                xmin = min( (ctrl_pts.x[j][i], ctrl_pts.x[j][i+1], ctrl_pts.x[j+1][i], ctrl_pts.x[j+1][i+1]) )
                ymax = max( (ctrl_pts.y[j][i], ctrl_pts.y[j][i+1], ctrl_pts.y[j+1][i], ctrl_pts.y[j+1][i+1]) )
                ymin = min( (ctrl_pts.y[j][i], ctrl_pts.y[j][i+1], ctrl_pts.y[j+1][i], ctrl_pts.y[j+1][i+1]) )
                self.t[j, i, 0, :] = [xmin, xmax, ymin, ymax]

def generate_user_config( src_dir, src_fname, src_fext, dst_dir, dst_cfg_fname):
    img_src = img_desc()
    img_src.read_encoded(src_dir, src_fname, src_fext)
    src_pts = ctrl_pts_desc(4, 5)
    src_pts.setup_fish_eye_ctrl_pts(img_src.cols(), img_src.rows())
    img_dst = img_desc(img_src.cols(), img_src.rows())
    dst_pts = ctrl_pts_desc(4,5)
    img_dst.read_binary(dst_dir, src_fname, ".bin")
    dst_pts.setup_grid_ctrl_pts(img_dst.cols(), img_dst.rows())
    out_dict = {
        "src":
        {
            "image_format": img_src.format_desc,
            "control_points":
            {}
        }
        # Here we are

    with open(os.path.join(dst_dir, dst_cfg_fname+".json"), 'w') as f:
        json.dump(out_dict, f, indent=4)

src_dir = "C:/HHWork/ImWarping/Data/Input/PyDefault/" 
src_fname = "NIR_2020-03-18-05-08-04-429"
src_fext = ".png"
dst_dir = "C:/HHWork/ImWarping/Data/Output/PyDefault/"

dst_config_fname = src_fname+"user_config"+".json"


fig,axes = plt.subplots(1,2)
img_src.fwrite(dst_dir, src_fname, ".bin")
src_tiles = tiles_desc(3, 4)
src_tiles.setup_with_ctrl_pts(src_pts)

img_src.show(axes[0])
src_pts.show(axes[0])

img_dst.show(axes[1])

img_dst.show(axes[1])
dst_pts.show(axes[1])

plt.ion()
plt.draw()
# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
plt.show(block=True)
# plt.pause(0.001)
