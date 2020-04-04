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

{
    "src":
    {
        "image":
        {
            "folder": "...",
            "name": "...",
            "ext": ".bin",
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
        "tiles"
        {
            "cols":4,
            "rows":3,
            "x0": [...],
            "x1": [...],
            "y0": [...],
            "y1": [...]
        },
    },
    "dst":
    {
        "image":
        {
            "folder": "...",
            "name": "...",
            "ext": ".bin",
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
        "tiles"
        {
            "cols":4,
            "rows":3,
            "x0": [...],
            "x1": [...],
            "y0": [...],
            "y1": [...]
        },
    },
    "proj_matrix"
    {
        "cols":4,
        "rows":3,
        "matrices":
        [ {3x3}, {3x3}, ...]
    }
}