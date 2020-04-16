import matplotlib.pyplot as plt
import numpy as np
import json
import os
from PIL import Image
from scipy.spatial import distance 
import math

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

# distance from (x0, y0) to line seg [(x1,y1) to (x2, y2)]
def distance_point_to_line_seg(x0, y0, x1, y1, x2, y2):
    return abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / math.sqrt((y2-y1)**2+(x2-x1)**2)

def Bierer_Neely_Algo(P, Q, Ppr, Qpr, X):
    u = np.divide( np.dot(np.subtract(X,P), np.subtract(Q,P)) , np.square(distance.euclidean(Q,P)) )
    v = np.divide( np.dot(np.subtract(X,P), perpendicular(np.subtract(Q,P))), distance.euclidean(Q,P) )

    Xpr = Ppr + np.multiply(u,np.subtract(Qpr, Ppr)) + np.divide(np.multiply(v, perpendicular(np.subtract(Qpr,Ppr))), distance.euclidean(Qpr, Ppr))

    d = distance_point_to_line_seg(Xpr[0], Xpr[1], Ppr[0], Ppr[1], Qpr[0], Qpr[1])
    
    return {"x": Xpr[0],
            "y": Xpr[1],
            "d": d }

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

src_landmark = {
    "x": [src_inf["NIR parameter"]["LM"][0]["x"], 
          src_inf["NIR parameter"]["LM"][1]["x"], 
          src_inf["NIR parameter"]["LM"][2]["x"],
          src_inf["NIR parameter"]["LM"][3]["x"],
          src_inf["NIR parameter"]["LM"][4]["x"]],
    "y": [src_inf["NIR parameter"]["LM"][0]["y"], 
          src_inf["NIR parameter"]["LM"][1]["y"], 
          src_inf["NIR parameter"]["LM"][2]["y"],
          src_inf["NIR parameter"]["LM"][3]["y"],
          src_inf["NIR parameter"]["LM"][4]["y"]] }

# Landmark location definition:
# Q0          Q1 (left eye, right eye)
#       P        (nose)
#   Q3     Q4    (left lip, right lip)
 
Q0 = (dst_landmark["x"][0], dst_landmark["y"][0])
Q1 = (dst_landmark["x"][1], dst_landmark["y"][1])
P = (dst_landmark["x"][2], dst_landmark["y"][2])
Q3 = (dst_landmark["x"][3], dst_landmark["y"][3])
Q4 = (dst_landmark["x"][4], dst_landmark["y"][4])

Qpr0 = (src_landmark["x"][0], src_landmark["y"][0])
Qpr1 = (src_landmark["x"][1], src_landmark["y"][1])
Ppr = (src_landmark["x"][2], src_landmark["y"][2])
Qpr3 = (src_landmark["x"][3], src_landmark["y"][3])
Qpr4 = (src_landmark["x"][4], src_landmark["y"][4])

xv, yv = np.meshgrid(np.linspace(0,120,6), np.linspace(0,120,6))

num_points = sum(len(xv) for x in xv)

Xs = { "x": xv.flatten(), 
       "y": yv.flatten() }

xpr = np.zeros(num_points)
ypr = np.zeros(num_points)

for i in range(num_points):
    X = (Xs["x"][i], Xs["y"][i])
    v0 = Bierer_Neely_Algo(P, Q0, Ppr, Qpr0, X)
    v1 = Bierer_Neely_Algo(P, Q1, Ppr, Qpr1, X)
    v2 = Bierer_Neely_Algo(P, Q3, Ppr, Qpr3, X)
    v3 = Bierer_Neely_Algo(P, Q4, Ppr, Qpr4, X)

    weight = (v0["d"], v1["d"], v2["d"], v3["d"])
    xpr[i] = sum( np.multiply( (v0["x"], v1["x"], v2["x"], v3["x"]), weight) ) / sum(weight)
    ypr[i] = sum( np.multiply( (v0["y"], v1["y"], v2["y"], v3["y"]), weight) ) / sum(weight)

Xprs = {"x": xpr,
        "y": ypr}

fig, axes = plt.subplots(1,2)

axes[0].imshow(dst_img, cmap="gray")
axes[0].set_title("Dst")
dst_lm_x = (Q0[0], Q1[0], P[0], Q3[0], Q4[0])
dst_lm_y = (Q0[1], Q1[1], P[1], Q3[1], Q4[1])
axes[0].plot(dst_lm_x, dst_lm_y, "ro")
axes[0].plot(Xs["x"], Xs["y"], "b*")

axes[1].imshow(src_img, cmap="gray")
src_lm_x = (Qpr0[0], Qpr1[0], Ppr[0], Qpr3[0], Qpr4[0])
src_lm_y = (Qpr0[1], Qpr1[1], Ppr[1], Qpr3[1], Qpr4[1])
axes[1].plot(src_lm_x, src_lm_y, "ro")
axes[1].plot(Xprs["x"], Xprs["y"], "g*")


plt_show(fig)



