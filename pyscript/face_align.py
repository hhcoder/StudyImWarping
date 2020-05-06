import matplotlib.pyplot as plt
import numpy as np
import json
import os
from PIL import Image
from scipy.spatial import distance 
import math
import result_disp
import config_gen as cfg
import cmdline
import result_gen
import matplotlib.image as pltimage
import util

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

def distance_point_to_point(x0, y0, x1, y1):
    return math.sqrt((x0-x1)**2+(y0-y1)**2)

def mid_point(x0, y0, x1, y1):
    return ((x0+x1)/2, (y0+y1)/2)

def euclidean_dist(P0, P1):
    return distance_point_to_point(P0[0], P0[1], P1[0], P1[1])

def Bierer_Neely_Algo(P, Q, Ppr, Qpr, X):
    u = np.divide( np.dot(np.subtract(X,P), np.subtract(Q,P)) , np.square(euclidean_dist(Q,P)) )
    v = np.divide( np.dot(np.subtract(X,P), perpendicular(np.subtract(Q,P))), euclidean_dist(Q,P) )

    t1 = np.multiply(u,np.subtract(Qpr, Ppr))
    t2 = np.divide(np.multiply(v, perpendicular(np.subtract(Qpr,Ppr))), euclidean_dist(Q, P))
    Xpr = Ppr + t1 + t2

    #d = distance_point_to_line_seg(X[0], X[1], P[0], P[1], Q[0], Q[1])
    #if u > 0 and u < 1:
    #    d = abs(v)
    #elif u <= 0:
    #    d = distance_point_to_point(P[0], P[1], X[0], X[1])
    #else:
    #    d = distance_point_to_point(Q[0], Q[1], X[0], X[1])

    d = distance_point_to_point(Q[0], Q[1], X[0], X[1])

    l = math.sqrt((P[0]-Q[0])**2 + (P[1]-Q[1])**2)
    pq_mid = mid_point(P[0], P[1], Q[0], Q[1])
    f = distance_point_to_point(pq_mid[0], pq_mid[1], X[0], X[1])
    c = distance_point_to_point(Q[0], Q[1], X[0], X[1])
    
    return {"x": Xpr[0],
            "y": Xpr[1],
            "d": d,
            "l": l,
            "f": f,
            "c": c}

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

def ctrl_pts_bierer_neely( dst_pts, src_landmark, dst_landmark):
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

    # xv, yv = np.meshgrid(np.linspace(0,dst_img_height,dst_tile_ver_count), np.linspace(0,dst_img_width,dst_tile_hor_count))
    xv = dst_pts["x"]
    yv = dst_pts["y"]
    (rows, cols) = xv.shape

    # num_points = sum(len(xv) for x in xv)

    xpr = np.zeros(xv.shape)
    ypr = np.zeros(xv.shape)

    for j in range(rows):
        for i in range(cols):
            X = (xv[j][i], yv[j][i])
            v0 = Bierer_Neely_Algo(P, Q0, Ppr, Qpr0, X)
            v1 = Bierer_Neely_Algo(P, Q1, Ppr, Qpr1, X)
            v2 = Bierer_Neely_Algo(P, Q3, Ppr, Qpr3, X)
            v3 = Bierer_Neely_Algo(P, Q4, Ppr, Qpr4, X)

            lx = (v0["x"], v1["x"], v2["x"], v3["x"])
            ly = (v0["y"], v1["y"], v2["y"], v3["y"])
            dist = (v0["d"], v1["d"], v2["d"], v3["d"])
            dist_scale = np.divide(np.subtract(dist, np.min(dist)), np.max(dist)-np.min(dist))
            farness = (v0["f"], v1["f"], v2["f"], v3["f"])
            farness_scale = np.divide(np.subtract(farness, np.min(farness)), np.max(farness)-np.min(farness))
            cloness = (v0["c"], v1["c"], v2["c"], v3["c"])
            cloness_scale = np.divide(np.subtract(cloness, np.min(cloness)), np.max(cloness)-np.min(cloness))
            length = (v0["l"], v1["l"], v2["l"], v3["l"])
            a = 0.1
            b = 2
            p = 0
            #weight = np.power(np.divide(np.power(length, p), np.add(a, dist)), b)
            # weight = np.exp(np.negative(np.power(farness_scale, 2) ))
            # weight = np.divide(np.max(farness), farness)
            # weight = np.power(np.divide(np.max(farness), farness), 4) #lip works in this one
            #weight = np.power(np.divide(np.max(farness_scale), farness_scale), 4) #lip works in this one
            # weight = np.power(np.divide(np.min(cloness), cloness), 4) 
            # weight = np.exp(np.negative(np.power(cloness_scale, 2) ))
            weight = np.exp(np.negative(np.divide(np.power(dist_scale, 2), 0.25))) #Almost working version
            # weight = (1, 1, 1, 1)
            # weight = np.power(np.divide(np.max(dist_scale), dist_scale), 4)
            # weight = np.power(np.divide(np.max(dist_scale), dist_scale), 4)
            # idx_min = np.argmax(weight)
            # val_min2 = np.max(np.delete(weight, idx_min))
            # idx_min2 = np.where(np.isclose(weight,val_min2))
            # xpr[j][i] = (weight[idx_min] * lx[idx_min] + weight[idx_min2] * lx[idx_min2]) / (weight[idx_min] + weight[idx_min2])
            # ypr[j][i] = (weight[idx_min] * ly[idx_min] + weight[idx_min2] * ly[idx_min2]) / (weight[idx_min] + weight[idx_min2])
            xpr[j][i] = sum( np.multiply(lx, weight) ) / sum(weight)
            ypr[j][i] = sum( np.multiply(ly, weight) ) / sum(weight)
            #xpr[j][i] = lx[idx_min]
            #ypr[j][i] = ly[idx_min]
            # xpr[j][i] = v3["x"]
            # ypr[j][i] = v3["y"]

    return { "x": xpr,
             "y": ypr }  

#src_dir = "../../Data/Input/FR-01/"
#src_inf_fname = "INF_2020-03-18-04-55-32-411"
#src_png_fname = "NIR_2020-03-18-04-55-32-411"
#src_png_ext = ".png"

#dst_dir = "../../Data/Output/FR-01/"

#src_dir = "../../Data/Input/FR-02/"
#src_inf_fname = "INF_2020-02-04_01-40-46"
#src_png_fname = "NIR_2020-02-04_01-40-46"
#src_png_ext = ".png"

#dst_dir = "../../Data/Output/FR-02/"

src_dir = "../../Data/Input/FR-03/"
src_inf_fname = "INF_20200307_163238"
src_png_fname = "IMG_20200307_163238"
src_png_ext = ".png"

dst_dir = "../../Data/Output/FR-03/"

src_img_fname = src_png_fname + "_src_img"
src_img_format = "raw8"

dst_img_width = 112 
dst_img_height = 112
dst_tile_hor_count = 7 
dst_tile_ver_count = 7 
dst_img_format = "raw8"

dst_img_fname = src_png_fname + "_warpped_img"

control_point_dim = (dst_tile_hor_count, dst_tile_ver_count)

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

user_config_fname = src_png_fname + "_user_config"
input_data_fname = src_png_fname + "_input_data"
driver_setting_fname = src_png_fname + "_driver_setting"
master_location_fname = src_png_fname + "_main"
dbg_setting_fname = src_png_fname + "_debug_setting"

# src_img = read_img(src_dir, src_png_fname, src_png_ext)

src_img_info = cfg.decode_image_to_binary_format(
    src_dir, src_png_fname, src_png_ext,
    dst_dir, src_img_fname, src_img_format)


dst_img_info = {
    "width": dst_img_width,
    "height": dst_img_height,
    "stride": dst_img_width,
    "format": dst_img_format }

dst_pts = ctrl_pts_rect_grid(
    dst_img_info["width"], dst_img_info["height"],
    control_point_dim[0], control_point_dim[1])

dst_landmark = {
    "x": np.array([38.2946, 73.5318, 56.0252, 41.5493, 70.7299]),
    "y": np.array([51.6963, 51.5014, 71.7366, 92.3655, 92.2041]) }

#dst_landmark["x"] = np.multiply(dst_landmark["x"], 480/112)
#dst_landmark["y"] = np.multiply(dst_landmark["y"], 480/112)

src_inf = read_inf(src_dir, src_inf_fname)
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

# dst_landmark["x"] = src_landmark["x"]
# dst_landmark["y"] = src_landmark["y"]

src_pts = ctrl_pts_bierer_neely(
    dst_pts,
    src_landmark,
    dst_landmark)

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

result = result_gen.read(dst_dir, master_location_fname)

fig,axes = plt.subplots(1,2)
axes[0].imshow(result["src"]["img"], cmap="gray")
axes[0].plot(result["src"]["x"], result["src"]["y"], "*y")
axes[0].plot(src_landmark["x"], src_landmark["y"], "*b")
axes[0].set_title("SRC")

axes[1].imshow(result["dst"]["img"], cmap="gray")
axes[1].plot(dst_landmark["x"], dst_landmark["y"], "*b")
axes[1].plot(result["dst"]["x"], result["dst"]["y"], "*y")
axes[1].set_title("DST")

util.plt_show()

fig.savefig(os.path.join(dst_dir, dbg_setting_fname+"_result_fig.png"))
pltimage.imsave(os.path.join(dst_dir,dbg_setting_fname+"_input_image.png"), result["src"]["img"], cmap='gray')
pltimage.imsave(os.path.join(dst_dir,dbg_setting_fname+"_output_image.png"), result["dst"]["img"], cmap='gray')
