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

# todo: use p0, p1 instead of x0y0, x1y1
def mid_point(x0, y0, x1, y1):
    return ((x0+x1)/2, (y0+y1)/2)

def find_mid_point(p0, p1):
    return ((p0[0]+p1[0])/2, (p0[1]+p1[1])/2)

def find_point_by_ratio(p0, p1, ratio_from_0):
    if ratio_from_0 > 1.0:
        return "None"
    return (p0[0]+(p1[0]-p0[0])*ratio_from_0, p0[1]+(p1[1]-p0[1])*ratio_from_0)

def find_ratio_01_12(p0, p1, p2):
    d01 = distance_point_to_point(p0[0], p0[1], p1[0], p1[1]);
    d12 = distance_point_to_point(p1[0], p1[1], p2[0], p2[1]);
    return d01 / (d01+d12)

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

    xv = dst_pts["x"]
    yv = dst_pts["y"]
    (rows, cols) = xv.shape

    xpr = np.zeros(xv.shape)
    ypr = np.zeros(yv.shape)

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
            weight = np.exp(np.negative(np.divide(np.power(cloness_scale, 2), 0.05))) #Almost working version
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
            xpr[j][i] = v3["x"]
            ypr[j][i] = v3["y"]

    return { "x": xpr,
             "y": ypr }  

def ctrl_pts_bierer_neely_adv( dst_pts, src_landmark, dst_landmark):
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

    xv = dst_pts["x"]
    yv = dst_pts["y"]
    (rows, cols) = xv.shape

    xpr = np.zeros(xv.shape)
    ypr = np.zeros(yv.shape)

    for j in range(rows):
        for i in range(cols):
            X = (xv[j][i], yv[j][i])
            v0 = Bierer_Neely_Algo(P, Q0, Ppr, Qpr0, X)
            v1 = Bierer_Neely_Algo(P, Q1, Ppr, Qpr1, X)
            v2 = Bierer_Neely_Algo(P, Q3, Ppr, Qpr3, X)
            v3 = Bierer_Neely_Algo(P, Q4, Ppr, Qpr4, X)
            v4 = Bierer_Neely_Algo(Q0, Q1, Qpr0, Qpr1, X)
            v5 = Bierer_Neely_Algo(Q0, Q3, Qpr0, Qpr3, X)
            v6 = Bierer_Neely_Algo(Q1, Q3, Qpr1, Qpr3, X)
            v7 = Bierer_Neely_Algo(Q3, Q4, Qpr3, Qpr4, X)

            lx = (v0["x"], v1["x"], v2["x"], v3["x"], v4["x"], v5["x"], v6["x"], v7["x"])
            ly = (v0["y"], v1["y"], v2["y"], v3["y"], v4["y"], v5["y"], v6["y"], v7["y"])
            dist = (v0["d"], v1["d"], v2["d"], v3["d"], v4["d"], v5["d"], v6["d"], v7["d"])
            dist_scale = np.divide(np.subtract(dist, np.min(dist)), np.max(dist)-np.min(dist))
            farness = (v0["f"], v1["f"], v2["f"], v3["f"], v4["f"], v5["f"], v6["f"], v7["f"])
            farness_scale = np.divide(np.subtract(farness, np.min(farness)), np.max(farness)-np.min(farness))
            weight = np.exp(np.negative(np.divide(np.power(farness_scale, 2), 0.2))) #Almost working version
            weight = (1,1,1,1,1,1,1,1)
            xpr[j][i] = sum( np.multiply(lx, weight) ) / sum(weight)
            ypr[j][i] = sum( np.multiply(ly, weight) ) / sum(weight)

    return { "x": xpr,
             "y": ypr }  

def find_ext_pt(p0, p1, ratio):
    return (p1[0] + (p1[0]-p0[0])*(ratio), p1[1] + (p1[1]-p0[1])*ratio )

def find_offset(p0, p1):
    return (p1[0]-p0[0], p1[1]-p0[1])

def add_offset(p0, offset):
    return (p0[0]+offset[0], p0[1]+offset[1])

def find_proj_from_C_to_AB(A,B,C):
    Ax=A[0]
    Ay=A[1]
    Bx=B[0]
    By=B[1]
    Cx=C[0]
    Cy=C[1]

    t = ((Cx-Ax)*(Bx-Ax)+(Cy-Ay)*(By-Ay)) / ((Bx-Ax)**2 + (By-Ay)**2)

    Dx = Ax + t * (Bx-Ax)
    Dy = Ay + t * (By-Ay)
    return (Dx, Dy)

def ctrl_pts_ht( dst_pts, src_landmark, dst_landmark):
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

    xv = dst_pts["x"]
    yv = dst_pts["y"]
    (rows, cols) = xv.shape

    if rows != 7 or cols !=7:
        print("HT algo works only for 7x7")
        return "None"

    xpr = np.zeros(xv.shape)
    ypr = np.zeros(yv.shape)

    p32 = Qpr0
    xpr[3][2] = p32[0]
    ypr[3][2] = p32[1]

    p34 = Qpr1
    xpr[3][4] = p34[0]
    ypr[3][4] = p34[1]

    p33 = mid_point(xpr[3][2], ypr[3][2], xpr[3][4], ypr[3][4])
    xpr[3][3] = p33[0]
    ypr[3][3] = p33[1]

    p31 = find_ext_pt(p33, p32, 1)
    (xpr[3][1], ypr[3][1]) = p31

    p30 = find_ext_pt(p32, p31, 1)
    (xpr[3][0], ypr[3][0]) = p30

    p35 = find_ext_pt(p33, p34, 1)
    (xpr[3][5], ypr[3][5]) = p35

    p36 = find_ext_pt(p34, p35, 1)
    (xpr[3][6], ypr[3][6]) = p36

    p53 = mid_point(Qpr3[0], Qpr3[1], Qpr4[0], Qpr4[1])
    (xpr[5][3],ypr[5][3]) = p53

    p52 = find_ext_pt(p53, Qpr3, 7/13)
    (xpr[5][2], ypr[5][2]) = p52

    p54 = find_ext_pt(p53, Qpr4, 7/13)
    (xpr[5][4], ypr[5][4]) = p54

    p51 = find_ext_pt(p53, p52, 1)
    (xpr[5][1], ypr[5][1]) = p51

    p50 = find_ext_pt(p52, p51, 1)
    (xpr[5][0], ypr[5][0]) = p50

    p55 = find_ext_pt(p53, p54, 1)
    (xpr[5][5], ypr[5][5]) = p55

    p56 = find_ext_pt(p54, p55, 1)
    (xpr[5][6], ypr[5][6]) = p56

    p43 = Ppr
    (xpr[4][3],ypr[4][3]) = p43

    of33_to_34 = find_offset(p33, p34)
    of33_to_32 = find_offset(p33, p32)
    of53_to_54 = find_offset(p53, p54)
    of53_to_52 = find_offset(p53, p52)

    # original method: find p42 from p32 and p52
    #r33to43 = find_ratio_01_12(p33, p43, p53)
    #p42 = find_point_by_ratio(p32, p52, r33to43)
    #(xpr[4][2], ypr[4][2]) = p42

    # original method: find p44 from p34 and p54
    #p44 = find_point_by_ratio(p34, p54, r33to43)
    #(xpr[4][4],ypr[4][4]) = p44
    # now we have the central 9 points ready

    # new method, find p42 from p33, p32 and p53, p53
    p42 = add_offset(p43, find_mid_point(of33_to_32, of53_to_52))
    (xpr[4][2], ypr[4][2]) = p42

    p44 = add_offset(p43, find_mid_point(of33_to_34, of53_to_54))
    (xpr[4][4],ypr[4][4]) = p44

    # Upper middle part
    p23 = find_ext_pt(p43, p33, 1)
    (xpr[2][3], ypr[2][3]) = p23
    p13 = find_ext_pt(p33, p23, 1)
    (xpr[1][3], ypr[1][3]) = p13
    p03 = find_ext_pt(p23, p13, 1)
    (xpr[0][3], ypr[0][3]) = p03

    # Upper right part


    p24 = add_offset(p23, of33_to_34)
    (xpr[2][4], ypr[2][4]) = p24
    p25 = add_offset(p24, of33_to_34)
    (xpr[2][5], ypr[2][5]) = p25
    p26 = add_offset(p25, of33_to_34)
    (xpr[2][6], ypr[2][6]) = p26

    p14 = add_offset(p13, of33_to_34)
    (xpr[1][4], ypr[1][4]) = p14
    p15 = add_offset(p14, of33_to_34)
    (xpr[1][5], ypr[1][5]) = p15
    p16 = add_offset(p15, of33_to_34)
    (xpr[1][6], ypr[1][6]) = p16

    p04 = add_offset(p03, of33_to_34)
    (xpr[0][4], ypr[0][4]) = p04
    p05 = add_offset(p04, of33_to_34)
    (xpr[0][5], ypr[0][5]) = p05
    p06 = add_offset(p05, of33_to_34)
    (xpr[0][6], ypr[0][6]) = p06

    p45 = find_ext_pt(p43, p44, 1)
    (xpr[4][5], ypr[4][5]) = p45
    p46 = find_ext_pt(p44, p45, 1)
    (xpr[4][6], ypr[4][6]) = p46

    p41 = find_ext_pt(p43, p42, 1)
    (xpr[4][1], ypr[4][1]) = p41
    p40 = find_ext_pt(p42, p41, 1)
    (xpr[4][0], ypr[4][0]) = p40

    # Upper left prt

    p22 = add_offset(p23, of33_to_32)
    (xpr[2][2], ypr[2][2]) = p22
    p21 = add_offset(p22, of33_to_32)
    (xpr[2][1], ypr[2][1]) = p21
    p20 = add_offset(p21, of33_to_32)
    (xpr[2][0], ypr[2][0]) = p20

    p12 = add_offset(p13, of33_to_32)
    (xpr[1][2], ypr[1][2]) = p12
    p11 = add_offset(p12, of33_to_32)
    (xpr[1][1], ypr[1][1]) = p11
    p10 = add_offset(p11, of33_to_32)
    (xpr[1][0], ypr[1][0]) = p10

    p02 = add_offset(p03, of33_to_32)
    (xpr[0][2], ypr[0][2]) = p02
    p01 = add_offset(p02, of33_to_32)
    (xpr[0][1], ypr[0][1]) = p01
    p00 = add_offset(p01, of33_to_32)
    (xpr[0][0], ypr[0][0]) = p00

    # Lower middle part
    p63 = find_ext_pt(p43, p53, 1)
    (xpr[6][3], ypr[6][3]) = p63

    # Lower right part

    p64 = add_offset(p63, of53_to_54)
    (xpr[6][4], ypr[6][4]) = p64
    p65 = add_offset(p64, of53_to_54)
    (xpr[6][5], ypr[6][5]) = p65
    p66 = add_offset(p65, of53_to_54)
    (xpr[6][6], ypr[6][6]) = p66

    # Lower left part
    p62 = add_offset(p63, of53_to_52)
    (xpr[6][2], ypr[6][2]) = p62
    p61 = add_offset(p62, of53_to_52)
    (xpr[6][1], ypr[6][1]) = p61
    p60 = add_offset(p61, of53_to_52)
    (xpr[6][0], ypr[6][0]) = p60

    return { "x": xpr,
             "y": ypr }  

def ctrl_pts_flat( dst_pts, src_landmark, dst_landmark):
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

    xv = dst_pts["x"]
    yv = dst_pts["y"]
    (rows, cols) = xv.shape

    xpr = np.zeros(xv.shape)
    ypr = np.zeros(yv.shape)

    xpr0 = Qpr0[0] - (Qpr1[0]-Qpr0[0]) * 1.8
    xpr1 = Qpr1[0] + (Qpr1[0]-Qpr0[0]) * 1.8
    ypr0 = Ppr[1] - (Ppr[1]-Qpr0[1]) * 4
    ypr1 = Ppr[1] + (Ppr[1]-Qpr0[1]) * 2

    delta_x = (xpr1-xpr0)/cols
    delta_y = (ypr1-ypr0)/rows

    for j in range(rows):
        for i in range(cols):
            xpr[j][i] = (xpr0 + delta_x * i)
            ypr[j][i] = (ypr0 + delta_y * j)

    return { "x": xpr,
             "y": ypr }  

#src_dir = "../../data/input/fr-01/"
#src_inf_fname = "inf_2020-03-18-04-55-32-411"
#src_png_fname = "nir_2020-03-18-04-55-32-411"
#src_png_ext = ".png"

#dst_dir = "../../data/output/fr-01/"

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

src_pts = ctrl_pts_ht(
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
