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

    def read_encoded(self, dir, fname, fext):
        fpath = os.path.join(dir, fname + fext)
        self.im = np.array(Image.open(fpath).convert("L"), np.intc)
        self.inf = {
            "width": self.im.shape[1], 
            "height": self.im.shape[0], 
            "format": "raw8" }
    
    def rows(self):
        return self.im.shape[0]

    def cols(self):
        return self.im.shape[1]

    def read_binary(self, dir, bin_name, bin_ext, json_name, json_ext):
        self.inf = json.load(open(os.path.join(dir, json_name+json_ext))) 
        np_fmt = img_desc.data_format[self.inf["format"]]
        b = np.fromfile(os.path.join(dir, bin_name+bin_ext), np_fmt) 
        self.im = np.reshape(b.astype(np.intc), (self.inf["height"], self.inf["width"]))

    def fwrite(self, dir, bin_name, bin_ext, json_name, json_ext): 
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(os.path.join(dir, bin_name+bin_ext), "wb") as f:
            self.im.astype(np.uint8).tofile(f)
        with open(os.path.join(dir, json_name+json_ext), 'w') as f:
            self.inf["folder"]=dir
            self.inf["name"]=bin_name
            self.inf["ext"]=bin_ext
            json.dump(self.inf, f, indent=4)

    def show(self, axes):
        axes.imshow(self.im, cmap="gray")

def distance_to_center(xloc, yloc, width_half, height_half):
    idx_dist = math.sqrt((xloc-width_half)**2 + (yloc-height_half)**2)
    diag_dist = math.sqrt(width_half**2 + height_half**2)
    dist =  idx_dist/diag_dist 
    return math.sqrt(math.cos( dist * math.pi / 4))

class ctrl_pts_desc:
    def __init__(self, cols, rows):
        self.x = np.ndarray((rows,cols),dtype=np.intc)
        self.y = np.ndarray((rows,cols),dtype=np.intc)
    
    def rows(self):
        return self.x.shape[0]
    
    def cols(self):
        return self.x.shape[1]

    def setup_fish_eye_ctrl_pts(self, img_width, img_height):
        yidx = np.linspace(0, img_height, num=self.rows()).astype(int)
        xidx = np.linspace(0, img_width, num=self.cols()).astype(int)
        n = 0
        for j in range(0, self.rows()):
            for i in range(0, self.cols()):
                factor = distance_to_center(xidx[i], yidx[j], img_width/2, img_height/2)
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


src_dir = "C:/HHWork/ImWarping/Data/Input/PyDefault/" 
src_fname = "NIR_2020-03-18-05-08-04-429"
src_fext = ".png"
dst_dir = "C:/HHWork/ImWarping/Data/Output/PyDefault/"

img_src = img_desc()

img_src.read_encoded(src_dir, src_fname, src_fext)
fig,axes = plt.subplots(1,2)
img_src.fwrite(dst_dir, src_fname, ".bin", src_fname, ".json")
src_pts = ctrl_pts_desc(4, 5)
src_pts.setup_fish_eye_ctrl_pts(img_src.cols(), img_src.rows())

img_src.show(axes[0])
src_pts.show(axes[0])

img_dst = img_desc()
img_dst.read_binary(dst_dir, src_fname, ".bin", src_fname, ".json")
img_dst.show(axes[1])
dst_pts = ctrl_pts_desc(4,5)
dst_pts.setup_grid_ctrl_pts(img_dst.cols(), img_dst.rows())

img_dst.show(axes[1])
dst_pts.show(axes[1])

plt.ion()
plt.draw()
# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
plt.show(block=True)
# plt.pause(0.001)
