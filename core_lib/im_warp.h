#pragma once

#include <cstdint>

class tile_gray8
{
    struct buf_gray8_t
    {
        uint8_t* buf;
    };

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

    struct proc_config 
    {
        buf_gray8_t src_buf; 
        dim2d_t src_dim;
        pixel_loc_t src_p00;
        pixel_loc_t src_p01;
        pixel_loc_t src_p10;
        pixel_loc_t src_p11;

        buf_gray8_t dst_buf; 
        dim2d_t dst_dim;
        pixel_loc_t dst_p00;
        pixel_loc_t dst_p01;
        pixel_loc_t dst_p10;
        pixel_loc_t dst_p11;
    };

    static void warp_proc(proc_config& setting)
    {
        // find_line_map
            //for each dst line
                // find the src line start and end
                // find the intermediat each point location

        // for each pixel on line
            // interpolate
    }
};

