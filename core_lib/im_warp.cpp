#include "nlohmann/json.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

template <typename T>
struct img_generic_t : std::vector<T>
{
    static const std::string none("None");
    //allocator
    img_gray8_t(const int in_width, const int in_height, const int in_stride, const std::string& bin_path = none)
        : width(in_width), height(in_height), stride(in_stride), std::vector<T>(in_height*in_stride)
    {
        if (bin_path != none)
            read(bin_path)
    }

    void read(const std::string& bin_path)
    {
        std::ofstream ifs(bin_path, std::ios::in | std::ios::binary);
        if (!ifs)
        {
            std::cout << "error open file: " << bin_path;
        }
        else
        {
            std::size_t buf_size = sizeof(T) * height * stride;
            ifs.read((char*)&(this->at[0]), buf_size);
        }
    }

    void write(const std::string& bin_path)
    {
        std::ofstream ofs(bin_path, std::ios::out | std::ios::binary);
        if (!ofs)
        {
            std::cout << "error open file: " << bin_path;
        }
        else
        {
            std::size_t buf_size = sizeof(T) * height * stride;
            ofs.write((char*)&(this->at[0]), buf_size);
        }
    }

    T* buf() { return &(this->at[0]); }

    int width;
    int height;
    int stride;
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

        std::string input_data_fpath = js["dir"].get<std::string>() + js["input_data"].get<std::string>();

        std::ifstream ifs_indata(input_data_fpath.c_str());
        nlohmann::json js_indata;

        ifs_indata >> js_indata;

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
        img_gray8_t

        for (int j = 0; j < kp_rows - 1; j++)
        {
            for (int i = 0; i < kp_cols - 1; i++)
            {
            }
        }
        
    }
    catch (std::exception const& e)
    {
        std::cout << "There was an error: " << e.what() << std::endl;
    }

    // read in driver_setting 
    // read in input_data


    return 0;
}