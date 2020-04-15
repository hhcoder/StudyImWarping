import matplotlib.pyplot as plt
import numpy as np
import json
import os
from PIL import Image
from scipy.spatial import distance 

# empty_image = np.ones((112, 112, 3), dtype=np.intc)

# standard_landmark = [30.2946, 51.6963, 65.5318, 51.5014, 48.0252, 71.7366, 33.5493, 92.3655, 62.7299, 92.2041]

# # standard_x = standard_landmark[-2::-2]
# # standard_y = standard_landmark[-1::-2]

# standard_x = standard_landmark[::2]
# standard_y = standard_landmark[1::2]

# fig,axes = plt.subplots()

# axes.imshow(empty_image, cmap="gray")
# axes.plot(standard_x, standard_y, 'r.')
# plt.ion()
# plt.draw()
# # manager = plt.get_current_fig_manager()
# # manager.full_screen_toggle()
# plt.show(block=True)
# plt.pause(0.1)

def plt_show(fig, do_block=True):
    plt.ion()
    plt.draw()
    plt.show(block=do_block)
    plt.pause(0.2)
    if not do_block:
        plt.close("all")

def read_inf(src_dir, inf_fname, inf_fext=".json"):
    return json.load(open(os.path.join(src_dir, inf_fname+inf_fext)))

def read_img(src_dir, src_fname, src_fext=".png"):
    fpath = os.path.join(src_dir, src_fname + src_fext)
    return np.array(Image.open(fpath).convert("L"), dtype=np.intc)

def display_landmark(axes, src_info):
    for i in src_info["NIR parameter"]["LM"]:
        axes.plot(i["x"], i["y"], "yo")
    
def plot_dot(axes, pt, color_format="o"):
    axes.plot(pt[0], pt[1], color_format)

def perpendicular(vec):
    return (-vec[1], vec[0])

def Bierer_Neely_Algo(P, Q, Ppr, Qpr, X):
    u = np.divide( np.dot(np.subtract(X,P), np.subtract(Q,P)) , np.square(distance.euclidean(Q,P)) )
    v = np.divide( np.dot(np.subtract(X,P), perpendicular(np.subtract(Q,P))), distance.euclidean(Q,P) )

    Xpr = Ppr + np.multiply(u,np.subtract(Qpr, Ppr)) + np.divide(np.multiply(v, perpendicular(np.subtract(Qpr,Ppr))), distance.euclidean(Qpr, Ppr))
    return Xpr

src_dir = "../../Data/Input/FR-01/"
src_inf_fname = "INF_2020-03-18-04-55-32-411"
src_img_fname = "NIR_2020-03-18-04-55-32-411"

src_inf = read_inf(src_dir, src_inf_fname)
src_img = read_img(src_dir, src_img_fname)

dst_img = 255 * np.ones((112,112), dtype=np.uint8)
dst_landmark = {
    "x": [30.2946, 65.5318, 48.0252, 33.5493, 62.7299],
    "y": [51.6963, 51.5014, 71.7366, 92.3655, 92.2041] 
}

Qpr = (src_inf["NIR parameter"]["LM"][0]["x"], src_inf["NIR parameter"]["LM"][0]["y"])
Ppr = (src_inf["NIR parameter"]["LM"][2]["x"], src_inf["NIR parameter"]["LM"][2]["y"])
 
Q = (dst_landmark["x"][0], dst_landmark["y"][0])
P = (dst_landmark["x"][2], dst_landmark["y"][2])
X = (0, 0)

# u = np.divide( np.dot(np.subtract(X,P), np.subtract(Q,P)) , np.square(distance.euclidean(Q,P)) )
# v = np.divide( np.dot(np.subtract(X,P), perpendicular(np.subtract(Q,P))), distance.euclidean(Q,P) )

# Xpr = Ppr + np.multiply(u,np.subtract(Qpr, Ppr)) + np.divide(np.multiply(v, perpendicular(np.subtract(Qpr,Ppr))), distance.euclidean(Qpr, Ppr))

Xpr = Bierer_Neely_Algo(P, Q, Ppr, Qpr, X)

fig, axes = plt.subplots(2,1)

axes[0].imshow(src_img, cmap="gray")
plot_dot(axes[0], Qpr, "ro")
plot_dot(axes[0], Ppr, "bo")

axes[1].imshow(dst_img, cmap="gray")
plot_dot(axes[1], Q, "ro")
plot_dot(axes[1], P, "bo")
plot_dot(axes[1], X, "go")

plot_dot(axes[0], Xpr, "r*")

plt_show(fig)
