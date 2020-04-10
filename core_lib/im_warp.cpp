#include "im_warp.h"
#include "nlohmann/json.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

template <typename T>
struct img_generic_t : std::vector<T>
{
    //allocator
    img_generic_t(const int in_width, const int in_height, const int in_stride, const std::string& bin_path = "None")
        : width(in_width), height(in_height), stride(in_stride), std::vector<T>(in_height*in_stride)
    {
        if (bin_path != "None")
            read(bin_path);
    }

    void read(const std::string& bin_path)
    {
        std::ifstream ifs(bin_path, std::ios::in | std::ios::binary);
        if (!ifs)
        {
            std::cerr << "error open file: " << bin_path;
        }
        else
        {
            std::size_t buf_size = sizeof(T) * height * stride;
            ifs.read((char*)this->data(), buf_size);
        }
    }

    void write(
        const std::string& dst_dir,
        const std::string& dst_name,
        const std::string& dst_bin_ext)
    {
        const std::string bin_path = dst_dir + dst_name + dst_bin_ext;

        std::ofstream ofs_bin(bin_path, std::ios::out | std::ios::binary);

        if (!ofs_bin)
        {
            std::cout << "error open file: " << bin_path;
            return;
        }

        std::size_t buf_size = sizeof(T) * height * stride;
        ofs_bin.write((char*)this->data(), buf_size);

        std::cout << "dst_im_path: " << bin_path << std::endl;
    }

    int width;
    int height;
    int stride;
};

using img_gray8_t = img_generic_t<uint8_t>;

struct dbg_info_t
{
    std::string dir;
    std::string fname;
    std::string fext;
};

static void warp_proc_gray8(
    const img_gray8_t& src_img, 
    const std::vector<std::vector<int>>& src_ctrl_pts_x,
    const std::vector<std::vector<int>>& src_ctrl_pts_y,
    const std::vector<std::vector<int>>& dst_ctrl_pts_x,
    const std::vector<std::vector<int>>& dst_ctrl_pts_y,
    img_gray8_t& dst_img,
    nlohmann::json& js_dbg_info)
{
    {
        dbg::jdump jd_ctrl_pts(
            js_dbg_info["dir"].get<std::string>(), 
            js_dbg_info["dumped_control_points"]["fname"].get<std::string>(), 
            js_dbg_info["dumped_control_points"]["fext"].get<std::string>());

        jd_ctrl_pts["src_x"] = src_ctrl_pts_x;
        jd_ctrl_pts["src_y"] = src_ctrl_pts_y;
        jd_ctrl_pts["dst_x"] = dst_ctrl_pts_x;
        jd_ctrl_pts["dst_y"] = dst_ctrl_pts_y;
    }

    for (std::size_t j = 0; j < src_ctrl_pts_x.size()-1; j++)
    {
        for (std::size_t i = 0; i < dst_ctrl_pts_x[0].size()-1; i++)
        {
            tile_gray8::proc_config_t in_config;

            in_config.src_buf = src_img.data();

            in_config.src_dim.width = src_img.width;
            in_config.src_dim.height = src_img.height;
            in_config.src_dim.stride = src_img.stride;

            in_config.src_p00.x = src_ctrl_pts_x[j][i];
            in_config.src_p00.y = src_ctrl_pts_y[j][i];
            in_config.src_p01.x = src_ctrl_pts_x[j+1][i];
            in_config.src_p01.y = src_ctrl_pts_y[j+1][i];
            in_config.src_p10.x = src_ctrl_pts_x[j][i+1];
            in_config.src_p10.y = src_ctrl_pts_y[j][i+1];
            in_config.src_p11.x = src_ctrl_pts_x[j+1][i+1];
            in_config.src_p11.y = src_ctrl_pts_y[j+1][i+1];

            in_config.dst_buf = dst_img.data();

            in_config.dst_dim.width = dst_img.width;
            in_config.dst_dim.height = dst_img.height;
            in_config.dst_dim.stride = dst_img.stride;

            in_config.dst_p00.x = dst_ctrl_pts_x[j][i];
            in_config.dst_p00.y = dst_ctrl_pts_y[j][i];
            in_config.dst_p01.x = dst_ctrl_pts_x[j+1][i];
            in_config.dst_p01.y = dst_ctrl_pts_y[j+1][i];
            in_config.dst_p10.x = dst_ctrl_pts_x[j][i+1];
            in_config.dst_p10.y = dst_ctrl_pts_y[j][i+1];
            in_config.dst_p11.x = dst_ctrl_pts_x[j+1][i+1];
            in_config.dst_p11.y = dst_ctrl_pts_y[j+1][i+1];

            using namespace std::literals;
            tile_gray8::dbg_config_t dbg_config;
            dbg_config.tile_x_idx_str = "xidx_"s + std::to_string(i) + "_"s;
            dbg_config.tile_y_idx_str = "yidx_"s + std::to_string(j);
            tile_gray8::warp_proc(in_config);
        }
    }
}


int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Error: Caller has to provide path information in json format!" << std::endl;
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

        std::string input_data_fpath = js["dir"].get<std::string>() + js["input_data"].get<std::string>();

        std::ifstream ifs_indata(input_data_fpath.c_str());
        nlohmann::json js_indata;

        ifs_indata >> js_indata;

        std::string dbg_setting_fpath = js["dir"].get<std::string>() + js["debug_setting"].get<std::string>();
        std::ifstream ifs_dbg(dbg_setting_fpath.c_str());

        nlohmann::json js_dbg_info;
        ifs_dbg >> js_dbg_info;

        // Decode the driver setting
        std::vector<std::vector<int>> src_ctrl_pts_x = js_drv["src"]["control_points"]["x"];
        std::vector<std::vector<int>> src_ctrl_pts_y = js_drv["src"]["control_points"]["y"];
        std::vector<std::vector<int>> dst_ctrl_pts_x = js_drv["dst"]["control_points"]["x"];
        std::vector<std::vector<int>> dst_ctrl_pts_y = js_drv["dst"]["control_points"]["y"];

        const int kp_rows = src_ctrl_pts_x.size();
        const int kp_cols = src_ctrl_pts_x[0].size();

        int src_im_width = js_drv["src"]["image_format"]["width"].get<int>();
        int src_im_height = js_drv["src"]["image_format"]["height"].get<int>();
        int src_im_stride = js_drv["src"]["image_format"]["stride"].get<int>();
        std::string src_im_format = js_drv["src"]["image_format"]["format"].get<std::string>();

        int dst_im_width = js_drv["dst"]["image_format"]["width"].get<int>();
        int dst_im_height = js_drv["dst"]["image_format"]["height"].get<int>();
        int dst_im_stride = js_drv["dst"]["image_format"]["stride"].get<int>();
        std::string dst_im_format = js_drv["dst"]["image_format"]["format"].get<std::string>();

        std::cout << "src_im_width: " << src_im_width << std::endl;
        std::cout << "src_im_height: " << src_im_height << std::endl;
        std::cout << "src_im_stride: " << src_im_stride << std::endl;
        std::cout << "src_im_format: " << src_im_format << std::endl;

        std::cout << "dst_im_width: " << dst_im_width << std::endl;
        std::cout << "dst_im_height: " << dst_im_height << std::endl;
        std::cout << "dst_im_stride: " << dst_im_stride << std::endl;
        std::cout << "dst_im_format: " << dst_im_format << std::endl;

        std::cout << "kp_rows: " << kp_rows << std::endl;
        std::cout << "kp_cols: " << kp_cols << std::endl;

        // Decode the input image
        const std::string src_im_path =
            js_indata["src"]["image_bin"]["dir"].get<std::string>() + 
            js_indata["src"]["image_bin"]["name"].get<std::string>() +
            js_indata["src"]["image_bin"]["bin_ext"].get<std::string>();

        std::cout << "src_im_path: " << src_im_path << std::endl;

        img_gray8_t src_img(src_im_width, src_im_height, src_im_stride, src_im_path);
        img_gray8_t dst_img(dst_im_width, dst_im_height, dst_im_stride);

        warp_proc_gray8(
            src_img, 
            src_ctrl_pts_x, src_ctrl_pts_y, dst_ctrl_pts_x, dst_ctrl_pts_y, 
            dst_img,
            js_dbg_info);

        dst_img.write(
            js_indata["dst"]["image_bin"]["dir"].get<std::string>(),
            js_indata["dst"]["image_bin"]["name"].get<std::string>(),
            js_indata["dst"]["image_bin"]["bin_ext"].get<std::string>());
    }
    catch (std::exception const& e)
    {
        std::cout << "There was an error: " << e.what() << std::endl;
    }

    // read in driver_setting 
    // read in input_data


    return 0;
}