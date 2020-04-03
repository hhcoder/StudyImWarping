what should the Json file look like?


{
    "input data":
    {
        "rgb image binary":
        {
            "width": 640,
            "height": 480,
            "stride": 640,
            "format": "rgb565",
            "data": [....],
            "binary location": [...]
        },
        "rgb image path":
        {
        
        },
        "landmarks":
        {
        }
    },
    "correlation results":
    {
        "left eye debug data"
        {
            "nir src image": {...},
            "nir proc image": {...}
        },
    }
    "processed image"
    {
    }
}

what should the code look like?

dbg::dbg_dump j = dbg::dbg_dump(folder_location, file_name);

j["session"] = dbg::ds(data_structure);
j["image_session"]["nir"] = dbg::img(nir_img);
j["image_session"]["nir"] = dbg::img(nir_img);
j["image_session"]["rgb"] = dbg::img(rgb_img);
j["p"] = dbg::array(p);
j["data_value"] = data_value;

