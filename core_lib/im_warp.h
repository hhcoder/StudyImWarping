#pragma once

#include <cstdint>
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

#include "dbg.h"

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

    struct dbg_config_t
    {
        std::string tile_x_idx_str;
        std::string tile_y_idx_str;
    };

    template <typename T>
    struct point_t
    {
        T x;
        T y;
    };

    //template <typename SrcType, typename DstType>
    //static DstType clamp(DstType& lower_bound, DstType& upper_bound, SrcType& value)
    //{
    //    SrcType src_lower_bound = lower_bound;
    //    SrcType src_upper_bound = upper_bound;
    //    return static_cast<DstType>(std::max(std::min(value, src_lower_bound), src_upper_bound));
    //}

    static uint8_t clamp_float_to_uint8(const float& v)
    {
        if (v > 255.0f)
            return 255;
        if (v < 0.0f)
            return 0;
        return static_cast<uint8_t>(std::round(v));
    }

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

        int dst_tile_height = (setting.dst_p01.y - setting.dst_p00.y + 1);
        int dst_tile_width = (setting.dst_p10.x - setting.dst_p00.x + 1);

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

        // Calculate start points
        std::vector<int> dst_start_points_x(dst_tile_height);
        std::vector<int> dst_start_points_y(dst_tile_height);

        std::vector<float> src_start_points_x(dst_tile_height);
        std::vector<float> src_start_points_y(dst_tile_height);

        float src_start_point_delta_x = (setting.src_p01.x - setting.src_p00.x) / static_cast<float>(dst_tile_height);
        float src_start_point_delta_y = (setting.src_p01.y - setting.src_p00.y) / static_cast<float>(dst_tile_height);

        for (int j=0; j<dst_tile_height; j++)
        {
            dst_start_points_x[j] = setting.dst_p00.x;
            dst_start_points_y[j] = setting.dst_p00.y + j;

            src_start_points_x[j] = setting.src_p00.x + src_start_point_delta_x * j;
            src_start_points_y[j] = setting.src_p00.y + src_start_point_delta_y * j;
        }

        src_start_points_x[dst_tile_height - 1] = static_cast<float>(setting.src_p01.x);
        src_start_points_y[dst_tile_height - 1] = static_cast<float>(setting.src_p01.y);

        // end points:
        //                [1,0] <- p10
        //                  |
        //                  |
        //                  |
        //                  |
        //                [1,1] <- p11

        // Calculate end points
        std::vector<int> dst_end_points_x(dst_tile_height);
        std::vector<int> dst_end_points_y(dst_tile_height);

        std::vector<float> src_end_points_x(dst_tile_height);
        std::vector<float> src_end_points_y(dst_tile_height);

        float src_end_point_delta_x = (setting.src_p11.x - setting.src_p10.x) / static_cast<float>(dst_tile_height);
        float src_end_point_delta_y = (setting.src_p11.y - setting.src_p10.y) / static_cast<float>(dst_tile_height);

        for (int j=0; j<dst_tile_height; j++)
        {
            dst_end_points_x[j] = setting.dst_p10.x;
            dst_end_points_y[j] = setting.dst_p10.y + j;

            src_end_points_x[j] = setting.src_p10.x + src_end_point_delta_x * j;
            src_end_points_y[j] = setting.src_p10.y + src_end_point_delta_y * j;
        }
        src_end_points_x[dst_tile_height - 1] = static_cast<float>(setting.src_p11.x);
        src_end_points_y[dst_tile_height - 1] = static_cast<float>(setting.src_p11.y);


        //{
        //    d[dbg_cfg.tile_x_idx_str + dbg_cfg.tile_y_idx_str]["src"]["start_points"]["x"] = src_start_points_x;
        //    d[dbg_cfg.tile_x_idx_str + dbg_cfg.tile_y_idx_str]["src"]["start_points"]["y"] = src_start_points_y;
        //    d[dbg_cfg.tile_x_idx_str + dbg_cfg.tile_y_idx_str]["src"]["end_points"]["x"] = src_end_points_x;
        //    d[dbg_cfg.tile_x_idx_str + dbg_cfg.tile_y_idx_str]["src"]["end_points"]["y"] = src_end_points_y;
        //    d[dbg_cfg.tile_x_idx_str + dbg_cfg.tile_y_idx_str]["dst"]["start_points"]["x"] = dst_start_points_x;
        //    d[dbg_cfg.tile_x_idx_str + dbg_cfg.tile_y_idx_str]["dst"]["start_points"]["y"] = dst_start_points_y;
        //    d[dbg_cfg.tile_x_idx_str + dbg_cfg.tile_y_idx_str]["dst"]["end_points"]["x"] = dst_end_points_x;
        //    d[dbg_cfg.tile_x_idx_str + dbg_cfg.tile_y_idx_str]["dst"]["end_points"]["y"] = dst_end_points_y;
        //}

        // for each start point to end point pair
        for (int j = 0; j < dst_tile_height; j++)
        {
            const float delta_x = (src_end_points_x[j] - src_start_points_x[j]) / static_cast<float>(dst_tile_width);
            const float delta_y = (src_end_points_y[j] - src_start_points_y[j]) / static_cast<float>(dst_tile_width);
                
            for (int i = 0; i < dst_tile_width; i++)
            {
                float src_x = src_start_points_x[j] + delta_x * i;
                float src_y = src_start_points_y[j] + delta_y * i;

                int dst_x = dst_start_points_x[j] + i;
                int dst_y = dst_start_points_y[j];

                interpolate_pixel(
                    setting.src_buf,
                    setting.src_dim,
                    src_x, src_y,
                    setting.dst_buf,
                    setting.dst_dim,
                    dst_x, dst_y);

            }
        }
    }
};

