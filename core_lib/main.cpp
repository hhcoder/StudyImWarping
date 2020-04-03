#include "nlohmann/json.hpp"
#include "im_warp.h"
#include "opencv2/opencv.hpp"
#include "dbg/dbg.h"
#include <fstream>
#include <iostream>
#include <string>

struct Gray8
{
    Gray8(const std::string& in_file_path)
        : buf(nullptr), rows(0), cols(0), file_path(in_file_path)
    {
        cv::Mat gray = cv::imread(file_path.c_str());
        rows = gray.rows;
        cols = gray.cols;
        auto area = rows * cols;
        buf = new uint8_t[area];

        for (auto i = 0; i < area; i++)
        {
            buf[i] = gray.data[i * 3];
        }
    }

    ~Gray8() { delete[] buf; }
    uint8_t* buf;
    int rows;
    int cols;
    std::string file_path;
};

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

        Gray8 gray_8(js["src"]["folder"].get<std::string>() +
            js["src"]["y_img"]["name"].get<std::string>() +
            js["src"]["y_img"]["ext"].get<std::string>());

        img_t src_img(gray_8.cols, gray_8.rows, gray_8.buf);

        ctrl_points_t src_ctrl_pts(
            js["src"]["ctrl_points"]["width"].get<int>(),
            js["src"]["ctrl_points"]["height"].get<int>());
        {
            auto a = js["src"]["ctrl_points"]["content"];
            int idx = 0;
            for (auto i : a)
            {
                src_ctrl_pts.set(idx, i);
                idx++;
            }
        }

        img_t dst_img(gray_8.cols, gray_8.rows);

        ctrl_points_t dst_ctrl_pts(
            js["dst"]["ctrl_points"]["width"].get<unsigned>(),
            js["dst"]["ctrl_points"]["height"].get<unsigned>());
        {
            auto a = js["dst"]["ctrl_points"]["content"];
            int idx = 0;
            for (auto i : a)
            {
                dst_ctrl_pts.set(idx, i);
                idx++;
            }
        }

        const std::string dst_img_fpath =
            js["dst"]["folder"].get<std::string>() +
            js["dst"]["y_img"]["name"].get<std::string>() +
            js["dst"]["y_img"]["ext"].get<std::string>();

        im_warp warping(src_img, src_ctrl_pts, dst_img, dst_ctrl_pts);

        {
            using namespace std::string_literals;

            dbg::jdump d(js["dst"]["folder"].get<std::string>(), "im_warp", ".json");

            d["dst"]["img"] = d.img(
                "result_gray"s,
                "raw8"s,
                dst_img.buf(),
                { dst_img.width, dst_img.height, dst_img.width, 1 });
        }
    }
    catch (std::exception const& e)
    {
        std::cout << "There was an error: " << e.what() << std::endl;
    }

    return 0;
}
