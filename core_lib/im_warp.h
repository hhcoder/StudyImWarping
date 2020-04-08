#pragma once

#include <cstdint>
#include <iostream>
#include <vector>
#include <utility>

struct tile_gray8
{
    struct dim2d_t
    {
        int width;
        int height;
        int stride;
    };

    struct pixel_loc_t
    {
        int x;
        int y;
    };


    // meaning of p00, p01, p10, p11
    //[0,0] ---------[1,0]
    //  |              |
    //  |              |
    //  |              |
    //  |              |
    //[0,1] ---------[1,1]

    struct proc_config_t
    {
        const uint8_t* src_buf; 
        dim2d_t src_dim;
        pixel_loc_t src_p00;
        pixel_loc_t src_p01;
        pixel_loc_t src_p10;
        pixel_loc_t src_p11;

        uint8_t* dst_buf; 
        dim2d_t dst_dim;
        pixel_loc_t dst_p00;
        pixel_loc_t dst_p01;
        pixel_loc_t dst_p10;
        pixel_loc_t dst_p11;
    };

    template <typename T>
    struct point_t
    {
        T x;
        T y;
    };

    static void interpolate_pixel(
        const uint8_t* src_buf,
        const dim2d_t& src_dim,
        float src_x, float src_y,
        uint8_t* dst_buf,
        const dim2d_t& dst_dim,
        int dst_x, int dst_y)
    {
    }


    static void warp_proc(proc_config_t& setting)
    {
        std::cout << std::endl;
        std::cout << "src p00: (" << setting.src_p00.x << "," << setting.src_p00.y << ")";
        std::cout << " p10: (" << setting.src_p10.x << "," << setting.src_p10.y << ")";
        std::cout << " p01: (" << setting.src_p01.x << "," << setting.src_p01.y << ")";
        std::cout << " p11: (" << setting.src_p11.x << "," << setting.src_p11.y << ")" << std::endl;

        std::cout << "dst p00: (" << setting.dst_p00.x << "," << setting.dst_p00.y << ")";
        std::cout << " p10: (" << setting.dst_p10.x << "," << setting.dst_p10.y << ")";
        std::cout << " p01: (" << setting.dst_p01.x << "," << setting.dst_p01.y << ")";
        std::cout << " p11: (" << setting.dst_p11.x << "," << setting.dst_p11.y << ")" << std::endl;

        int dst_tile_height = (setting.dst_p01.y - setting.dst_p00.y);
        int dst_tile_width = (setting.dst_p10.x - setting.dst_p00.x);

        // Calculate start points
        std::vector<int> dst_start_points_x(dst_tile_height);
        std::vector<int> dst_start_points_y(dst_tile_height);

        std::vector<float> src_start_points_x(dst_tile_height);
        std::vector<float> src_start_points_y(dst_tile_height);

        float start_point_delta_x = (setting.src_p01.x - setting.src_p00.x) / static_cast<float>(dst_tile_height);
        float start_point_delta_y = (setting.src_p01.y - setting.src_p00.y) / static_cast<float>(dst_tile_height);

        for (int j=0; j<dst_tile_height; j++)
        {
            dst_start_points_x[j] = setting.dst_p00.x;
            dst_start_points_y[j] = setting.dst_p00.y + j;

            src_start_points_x[j] = setting.src_p00.x + start_point_delta_x * j;
            src_start_points_y[j] = setting.src_p00.y + start_point_delta_y * j;
        }

        src_start_points_x[dst_tile_height - 1] = static_cast<float>(setting.src_p11.x);
        src_start_points_y[dst_tile_height - 1] = static_cast<float>(setting.src_p11.y);

        // Calculate end points
        std::vector<int> dst_end_points_x(dst_tile_height);
        std::vector<int> dst_end_points_y(dst_tile_height);

        std::vector<float> src_end_points_x(dst_tile_height);
        std::vector<float> src_end_points_y(dst_tile_height);

        float end_point_delta_x = (setting.src_p11.x - setting.src_p10.x) / static_cast<float>(dst_tile_height);
        float end_point_delta_y = (setting.src_p11.y - setting.src_p10.y) / static_cast<float>(dst_tile_height);

        for (int j=0; j<dst_tile_height; j++)
        {
            dst_end_points_x[j] = setting.dst_p10.x;
            dst_end_points_y[j] = setting.dst_p10.y + j;

            src_end_points_x[j] = setting.src_p10.x + end_point_delta_x * j;
            src_end_points_y[j] = setting.src_p10.y + end_point_delta_y * j;
        }

        src_end_points_x[dst_tile_height - 1] = static_cast<float>(setting.src_p11.x);
        src_end_points_y[dst_tile_height - 1] = static_cast<float>(setting.src_p11.y);

        // for each start point to end point pair
        std::vector<int> dst_line_x(dst_tile_width);
        std::vector<int> dst_line_y(dst_tile_width);

        std::vector<float> src_line_x(dst_tile_width);
        std::vector<float> src_line_y(dst_tile_width);

        for (int j = 0; j < dst_tile_height; j++)
        {
            for (int i = 0; i < dst_tile_width; i++)
            {
                int dst_x = dst_start_points_x[j] + i;
                int dst_y = dst_start_points_y[j];

                const float delta_x = (src_end_points_x[j] - src_start_points_x[j]) / static_cast<float>(dst_tile_width);
                const float delta_y = (src_end_points_y[j] - src_start_points_y[j]) / static_cast<float>(dst_tile_width);
                
                float src_x = src_start_points_x[j] + delta_x * i;
                float src_y = src_start_points_y[j] + delta_y * i;

                // nearest as first implementation
                int nearest_src_x = static_cast<int>(std::round(src_x));
                int nearest_src_y = static_cast<int>(std::round(src_y));

                setting.dst_buf[dst_y * setting.dst_dim.stride + dst_x] = 
                    setting.src_buf[nearest_src_y * setting.src_dim.stride + nearest_src_x];
            }
        }
                // find the intermediat each point location

        // for each pixel on line
            // interpolate
    }
};

