 GenerateDescModule
 . read in image, with provided setting
 . Generate key points
 . based on key points, calculate tiles of src and dst
 . filled in all informations

class ImageDesc
def CtrlPtsDesc:
def TilesDesc:
def ProjMatrixDesc:

DisplayModule
. Display image
. display key points (from json file)
. display Tiles (from json file)
. list proj matrix?


# System Design

{Calibration Tool}   -\
{User Interface Tool} --> [user_config.json] -> {Convert Tool} -> [driver_setting.json] -> {C Implementation} -> [output_data.json]
                      |                                                                           ^
                      |-->[input_data.json] ------------------------------------------------------|
                                                                                                  |
                                                                                            [output_config.json]

## user_config.json
{
    "src":
    {
        "image_format":
        {
            "width": 640,
            "height": 480,
            "format": "gray8"
        },
        "control_points":
        {
            "cols":5,
            "rows":4,
            "x": [...],
            "y": [...]
        }
    },
    "dst":
    {
        "image_format":
        {
            "width": 640,
            "height": 480,
            "format": "gray8"
        },
        "control_points":
        {
            "cols":5,
            "rows":4,
            "x": [...],
            "y": [...]
        },
    }
}

## input_data.json
{
    "src":
    {
        "image file":
        {
            "folder": "C:/HHWork/ImWarping/Data/Input/PyDefault/"
            "name": "NIR_2020-03-18-05-08-04-429",
            "bin_ext": ".bin",
            "json_ext": ".json",
            "width": 640,
            "height": 480,
            "format": "gray8"
        },
    },
    "dst":
    {
        "image file":
        {
            "folder": "C:/HHWork/ImWarping/Data/Output/PyDefault/"
            "name": "NIR_2020-03-18-05-08-04-429",
            "bin_ext": ".bin",
            "json_ext": ".json",
            "width": 640,
            "height": 480,
            "format": "gray8"
        },
    }
}

## driver_setting.json
{
    "cols":4,
    "rows":3,
    "src_tiles":
    {
        "x0": [...],
        "x1": [...],
        "y0": [...],
        "y1": [...]
    },
    "proj_matrix":
        [ {3x3}, {3x3}, ...],
    "dst_tiles":
    {
        "x0": [...],
        "x1": [...],
        "y0": [...],
        "y1": [...]
    }
}

# FR part

## Bierer-Neely Algorithm

## Mixing between points

## Flow

Goal:
for each point in DST Xs:[X0, X1, ..., Xn]
find SRC Xprs:[Xpr0, Xpr1, ..., Xprn]

1) define 4 lines (P0-P2), (P1-P2), (P3-P2), (P4-P2)

2) for each point, we calculate the 
    4 Xpr, relative to line segment (P0-P2), (P1-P2), (P3-P2), (P4-P2)
    4 Dpr, relative to line segment (P0-P2), (P1-P2), (P3-P2), (P4-P2)

3) combine the 4 Xpr and 4 Dpr, find the final Xpr 