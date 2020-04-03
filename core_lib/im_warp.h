#pragma once

#include <utility>
#include <vector>
#include <array>
#include <cstdint>
#include <cmath>

// most likely we'll change float into Q number

using pixel_t = uint8_t;

//Generic 2D buffer
template <typename T>
struct g2d_buf_t : std::vector<T>
{
    g2d_buf_t(std::size_t x_count, std::size_t y_count, const T* src = nullptr )
        : std::vector<T>(x_count*y_count),
          width(x_count), height(y_count)
    { 
        if (src!=nullptr)
            memcpy(buf(), src, sizeof(T) * this->size());
    }

    decltype(auto) get(unsigned x, unsigned y) { return this->at(offset(x,y)); }
    template <typename InType>
    void set(unsigned x, unsigned y, const InType& in_v) { this->at(offset(x,y)) = static_cast<T>(in_v); }
    template <typename InType>
    void set(unsigned idx, const InType& in_v) { this->at(idx) = static_cast<T>(in_v); }
    T* buf() { return &(this->at(0)); }
    inline std::size_t offset(unsigned x, unsigned y) { return y * width + x; }
    std::size_t width;
    std::size_t height;
};

template <typename T>
using point_t = std::pair<T, T>;

template <typename T>
struct rect_t : std::vector<T>
{
    //perfect forwarding
    rect_t()
        : std::vector(4)
    {
    }
};

using ctrl_points_t = g2d_buf_t<point_t<unsigned>>;

using img_t = g2d_buf_t<pixel_t>;

using tile_loc_t = g2d_buf_t<rect_t<unsigned>>;

struct warp_unit 
{
    warp_unit(
            img_t& in_img,
            ctrl_points_t& in_ctrl_pts)
        : img(in_img), 
          ctrl_pts(in_ctrl_pts), 
          tiles(ctrl_pts.width-1, ctrl_pts.height-1)
    { 
        for (int j = 0; j < tiles.height; j++)
        {
            for (int i = 0; i < tiles.width; i++)
            {
                point_t<unsigned> p00 = ctrl_pts.get(i, j);
                point_t<unsigned> p01 = ctrl_pts.get(i+1, j);
                point_t<unsigned> p10 = ctrl_pts.get(i, j+1);
                point_t<unsigned> p11 = ctrl_pts.get(i+1, j+1);

                unsigned x_min = min4(p00.first, p01.first, p10.first, p11.first);
                unsigned x_max = max4(p00.first, p01.first, p10.first, p11.first);
                unsigned y_min = min4(p00.second, p01.second, p10.second, p11.second);
                unsigned y_max = max4(p00.second, p01.second, p10.second, p11.second);

                tiles.set(i, j, { x_min, y_min, x_max - x_min + 1, y_max - y_min + 1 });
            }
        }
    }

    template <typename T>
    static T min4(const T& t1, const T& t2, const T& t3, const T& t4)
    { return std::min(std::min(t1, t2), std::min(t3, t4)); }

    template <typename T>
    static T max4(const T& t1, const T& t2, const T& t3, const T& t4)
    { return std::max(std::max(t1, t2), std::max(t3, t4)); }

    img_t& img;
    ctrl_points_t& ctrl_pts;
    tile_loc_t tiles;
};

struct mapping_functor
{
    mapping_functor(
            const std::array<point_t<int>, 4>& src, 
            const std::array<point_t<int>, 4>& dst)
    {
    }

    point_t<float> operator()(const int dst_x, const int dst_y) 
    { 
        return std::make_pair(
            static_cast<float>(dst_x), 
            static_cast<float>(dst_y));
    }
};

struct interp_functor
{
    interp_functor(img_t& in_src)
        : src(in_src) {}

    uint8_t operator()(const float src_x, const float src_y)
    {
        return src.get(
            static_cast<int>(std::floor(src_x)), 
            static_cast<int>(std::floor(src_y)));
    }

    img_t& src;
};

class im_warp
{
public:
    im_warp(
            img_t& in_src_img, 
            ctrl_points_t& in_src_ctrl_pts,
            img_t& in_dst_img,
            ctrl_points_t& in_dst_ctrl_pts)
        : src(in_src_img, in_src_ctrl_pts),
          dst(in_dst_img, in_dst_ctrl_pts)
    { }

private:
    warp_unit src;
    warp_unit dst;

    void setup_src()
    {
    }

    void setup_dst()
    {
    }

    void setup()
    {
        //setup interpolate functor
        //setup mapping functor
    }

    void process()
    {
        // for each tile, process
    }

    void process_tile()
    {
        // for each pixel
        //     calc_mapping from dst to src
        //     interpolate from src to dst
    }

    template <typename MappingFunctorType>
    static point_t<float> calc_mapping(const MappingFunctorType& mapping_fxn, int dst_x, int dst_y)
    {
        return mapping_fxn(dst_x, dst_y);
    }

    template <typename InterpFunctorType>
    static pixel_t interpolate(
            const InterpFunctorType& interp_fxn,
            const float src_x,
            const float src_y)
    {
        return interp_fxn(src_x, src_y);
    }
};
