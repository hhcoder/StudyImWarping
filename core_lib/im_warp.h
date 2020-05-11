#pragma once

#include <cstdint>
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

#include "dbg.h"

struct tile_gray8
{
    static uint8_t clamp_float_to_uint8(const float& v)
    {
        if (v > 255.0f)
            return 255;
        if (v < 0.0f)
            return 0;
        return static_cast<uint8_t>(std::round(v));
    }

    struct dim2d_t
    {
        int width;
        int height;
        int stride;
    };

    template <typename T>
    struct point_t
    {
        T operator/(const float& b) const { return { x / b, y / b }; }
        point_t<T> operator-(const point_t<T>& b) const { return { this->x - b.x, this->y - b.y }; }
        T x;
        T y;
    };

    using pixel_loc_t = point_t<int>;

    //struct pixel_loc_t
    //{
    //    int x;
    //    int y;
    //};

    struct tile_proc_user_config_t
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

    static void interpolate_pixel(
        const uint8_t* src_buf,
        const dim2d_t& src_dim,
        float src_x, float src_y,
        uint8_t* dst_buf,
        const dim2d_t& dst_dim,
        int dst_x, int dst_y)
    {
        if (dst_y >= dst_dim.height || dst_x >= dst_dim.width)
            std::cout << "dst_y: " << dst_y << ", dst_x: " << dst_x << std::endl;

        // nearest implementation
#if 0
        int nearest_src_x = static_cast<int>(std::round(src_x));
        int nearest_src_y = static_cast<int>(std::round(src_y));

        if (nearest_src_y >= src_dim.height || nearest_src_x >= src_dim.width)
            std::cout << "nearest_src_y: " << nearest_src_y << ", nearest_src_x: " << nearest_src_x << std::endl;

        dst_buf[dst_y * dst_dim.stride + dst_x] = 
            src_buf[nearest_src_y * src_dim.stride + nearest_src_x];
#endif

#if 1
        // bilinear interpolation, float implementation
        int src_x_0 = static_cast<int>(std::floor(src_x));
        int src_x_1 = src_x_0 + 1;
        int src_y_0 = static_cast<int>(std::floor(src_y));
        int src_y_1 = src_y_0 + 1;

        float hor_left = src_x - src_x_0;
        float hor_right = 1.0f - hor_left;

        float hor_mix_top = hor_left  * src_buf[src_x_1 + src_y_0 * src_dim.stride] + 
                            hor_right * src_buf[src_x_0 + src_y_0 * src_dim.stride];

        float hor_mix_bottom = hor_left *  src_buf[src_x_1 + src_y_1 * src_dim.stride] +
                               hor_right * src_buf[src_x_0 + src_y_1 * src_dim.stride];

        float ver_top = src_y - src_y_0;
        float ver_bott = 1.0f - ver_top;

        float v = ver_top * hor_mix_bottom + ver_bott * hor_mix_top;

        dst_buf[dst_y * dst_dim.stride + dst_x] = clamp_float_to_uint8(v);
#endif
    }

    // meaning of p00, p01, p10, p11
    //[0,0] ---------[1,0]
    //  |              |
    //  |              |
    //  |              |
    //  |              |
    //[0,1] ---------[1,1]

    // start points:
    //[0,0]  <-p00
    //  |   
    //  |   
    //  |   
    //  |   
    //[0,1]  <- p01

    // end points:
    //                [1,0] <- p10
    //                  |
    //                  |
    //                  |
    //                  |
    //                [1,1] <- p11

    struct tile_proc_driver_setting_t
    {
        tile_proc_driver_setting_t(const tile_proc_user_config_t& user_config)
            : dst_tile_height(user_config.dst_p01.y - user_config.dst_p00.y),
              dst_tile_width(user_config.dst_p10.x - user_config.dst_p00.x),
              dst_start_points(dst_tile_height),
              src_start_points(dst_tile_height),
              src_line_delta(dst_tile_height)
        {
            const point_t<float> src_start_point_delta =
                { (user_config.src_p01.x - user_config.src_p00.x) / static_cast<float>(dst_tile_height),
                  (user_config.src_p01.y - user_config.src_p00.y) / static_cast<float>(dst_tile_height) };

            const point_t<float> src_end_point_delta =
                { (user_config.src_p11.x - user_config.src_p10.x) / static_cast<float>(dst_tile_height),
                  (user_config.src_p11.y - user_config.src_p10.y) / static_cast<float>(dst_tile_height) };

            for (int j = 0; j < dst_tile_height; j++)
            {
                dst_start_points[j] = { user_config.dst_p00.x, user_config.dst_p00.y + j };

                src_start_points[j] =
                    { user_config.src_p00.x + src_start_point_delta.x * j,
                      user_config.src_p00.y + src_start_point_delta.y * j };

                point_t<float> src_end_points =
                    { user_config.src_p10.x + src_end_point_delta.x * j,
                      user_config.src_p10.y + src_end_point_delta.y * j };

                src_line_delta[j].x = (src_end_points.x - src_start_points[j].x) / static_cast<float>(dst_tile_width);
                src_line_delta[j].y = (src_end_points.y - src_start_points[j].y) / static_cast<float>(dst_tile_width);
            }
        }

        int dst_tile_height;
        int dst_tile_width;

        std::vector<point_t<int>> dst_start_points;

        std::vector<point_t<float>> src_start_points;

        std::vector<point_t<float>> src_line_delta;
    };

    static void tile_warp_proc(tile_proc_user_config_t& user_config)
    {
        std::cout << std::endl;
        std::cout << "src p00: (" << user_config.src_p00.x << "," << user_config.src_p00.y << ")";
        std::cout << " p10: (" << user_config.src_p10.x << "," << user_config.src_p10.y << ")";
        std::cout << " p01: (" << user_config.src_p01.x << "," << user_config.src_p01.y << ")";
        std::cout << " p11: (" << user_config.src_p11.x << "," << user_config.src_p11.y << ")" << std::endl;

        std::cout << "dst p00: (" << user_config.dst_p00.x << "," << user_config.dst_p00.y << ")";
        std::cout << " p10: (" << user_config.dst_p10.x << "," << user_config.dst_p10.y << ")";
        std::cout << " p01: (" << user_config.dst_p01.x << "," << user_config.dst_p01.y << ")";
        std::cout << " p11: (" << user_config.dst_p11.x << "," << user_config.dst_p11.y << ")" << std::endl;

        // User config to driver setting
        tile_proc_driver_setting_t driver_setting(user_config);

        for (int j = 0; j < driver_setting.dst_tile_height; j++)
        {
            for (int i = 0; i < driver_setting.dst_tile_width; i++)
            {
                float src_x = driver_setting.src_start_points[j].x + driver_setting.src_line_delta[j].x * i; 
                float src_y = driver_setting.src_start_points[j].y + driver_setting.src_line_delta[j].y * i;

                int dst_x = driver_setting.dst_start_points[j].x + i;
                int dst_y = driver_setting.dst_start_points[j].y;

                interpolate_pixel(
                    user_config.src_buf,
                    user_config.src_dim,
                    src_x, src_y,
                    user_config.dst_buf,
                    user_config.dst_dim,
                    dst_x, dst_y);

            }
        }
    }
};

