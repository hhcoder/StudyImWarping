import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import json
import os

class img_loader:
    data_format = {'raw8': np.uint8,
                    'float': np.float}

    def __init__(self, cols=None, rows=None):
        if cols is not None and rows is not None:
            self.im = np.empty((cols, rows), dtype=np.intc)
            self.format_desc = {
                "width": self.im.shape[1], 
                "height": self.im.shape[0], 
                "format": "raw8" }

    def read_encoded(self, dir, fname, fext):
        fpath = os.path.join(dir, fname + fext)
        self.im = np.array(Image.open(fpath).convert("L"), dtype=np.intc)
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
        np_fmt = img_loader.data_format[self.format["format"]]
        b = np.fromfile(os.path.join(dir, fname+bin_ext), np_fmt) 
        self.im = np.reshape(b.astype(np.intc), (self.format["height"], self.format["width"]))

    # def fwrite(self, dir, fname, bin_ext, json_ext = ".json"): 
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     with open(os.path.join(dir, fname+bin_ext), "wb") as f:
    #         self.im.astype(np.uint8).tofile(f)
    #     with open(os.path.join(dir, fname+json_ext), 'w') as f:
    #         self.file_desc = {
    #             "folder": dir,
    #             "name": fname,
    #             "bin_ext": bin_ext,
    #             "json_ext": ".json",
    #             "width": self.cols(),
    #             "height": self.rows(),
    #             "format": "raw8"}
    #         json.dump(self.file_desc, f, indent=4)

    def show(self, axes):
        axes.imshow(self.im, cmap="gray")

def plt_show(full_screen = True):
    plt.ion()
    if full_screen:
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=False)
