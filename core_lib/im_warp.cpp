#include "nlohmann/json.hpp"
#include <vector>
#include <iostream>
#include <fstream>

struct img_gray8_t : std::vector<uint8_t>
{
    //allocator
    //loader
    //writer
};

static void warp_proc_gray8(img_gray8_t& src_img, img_gray8_t& dst_img, nlohmann::json js)
{
    // for each tile, call tile process 
}


int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Error: Caller has to provide in path information in json format" << std::endl;
        return 1;
    }

    const char* in_json_path = argv[1];

    try
    {
        std::ifstream ifs(in_json_path);
        nlohmann::json js;

        ifs >> js;

        std::string driver_setting_fpath = js["dir"].get<std::string>() + js["driver_setting"].get<std::string>();
        std::ifstream ifs_drv(driver_setting_fpath.c_str());

        nlohmann::json js_drv;
        ifs_drv >> js_drv;

        std::vector<std::vector<int>> src_ctrl_pts_x = js_drv["src"]["control_points"]["x"];
        std::vector<std::vector<int>> src_ctrl_pts_y = js_drv["src"]["control_points"]["y"];
        std::vector<std::vector<int>> dst_ctrl_pts_x = js_drv["dst"]["control_points"]["x"];
        std::vector<std::vector<int>> dst_ctrl_pts_y = js_drv["dst"]["control_points"]["y"];

        std::cout << "rows: " << src_ctrl_pts_x.size() << std::endl;
        std::cout << "cols: " << src_ctrl_pts_x[0].size() << std::endl;
        //for (auto it = src_ctrl_pts_x.begin(); it != src_ctrl_pts_x.end(); it++)
        //{
        //    std::vector<int> v = *it;
        //    std::cout << "(" << v[0] << "," << v[1] << "," << v[2] << "," << v[3] << ")";  
        //}
        //std::cout << std::endl;
        
        int src_im_width = js_drv["src"]["image_format"]["width"].get<int>();
        int src_im_height = js_drv["src"]["image_format"]["height"].get<int>();

        std::string input_data_fpath = js["dir"].get<std::string>() + js["input_data"].get<std::string>();

        std::cout << "im_width: " << src_im_width << std::endl;
        std::cout << "im_height: " << src_im_height << std::endl;
        std::ifstream ifs_indata(input_data_fpath.c_str());
        nlohmann::json js_indata;

        ifs_indata >> js_indata;

    }
    catch (std::exception const& e)
    {
        std::cout << "There was an error: " << e.what() << std::endl;
    }

    // read in driver_setting 
    // read in input_data


    return 0;
}