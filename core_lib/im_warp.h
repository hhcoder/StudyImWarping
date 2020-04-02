#pragma once

#include <pair>
#include <vector>
#include <cstdint>

// most likely we'll change float into Q number

using pixel_t = uint8_t;

//Generic 2D buffer
template <typename T>
struct g2d_buf_t : std::vector<T>
{
    g2d_buf_t(int x_count, int y_count)
        : std::vector<std::vector<T>>(y_count, vector<T>(x_count)) { }

    void set(int x, int y, const T& in_v) { this->at(y).at(x) = in_v; }
    const T get(int x, int y) const { return this->at(y).at(x); }
    std::vector<std::vector<T>>::size_type height() const { return this->size(); } 
    std::vector<T>::size_type width() const { return this->at(0).size();
};

template <typename CoordType>
using point_t = std::pair<CoordType, CoordType>;

template <typename T>
using rect_t = std::array<T, 4>;

using ctrl_points_t = g2d_buf_t<point_t<int>>;

using img_t = g2d_buf_t<pixel_t>;

using tile_loc_t = std::vector<rect_t<int>>;

struct warp_unit 
{
    warp_unit(
            img_t& in_img,
            ctrl_points_t& in_ctrl_pts)
        : img(in_img), 
          ctrl_pts(in_ctrl_pts), 
          tiles(ctrl_pts.width()-1*ctrl_pts.height()-1)
    { 
        //calculate tile location
    }
    img_t& img;
    ctrl_points_t& ctrl_pts;
    tile_loc_t tiles;
    inline int im_width(){ return 0; }
    inline int im_height(){ return 0; }
    inline int ctrl_x_count(){ return 0;}
    inline int ctrl_y_count(){ return 0;}
};

// using location_mapping = g2d_buf_t<point_t<float>>;

struct mapping_functor
{
    mapping_functor(
            const std::array<point_t<int>, 4>& src, 
            const std::array<point_t<int>, 4>& dst)
    {
    }

    point_t<float> operator()(const int dst_x, const int dst_y) 
    { 
        return std::make_pair<float>(dst_x, dst_y);
    }
};

struct interp_functor
{
    uint8_t operator()(const float src_x, const float src_y)
    {
        return psrc->get(std::floor(src_x), std::floor(src_y));
    }

    img_t* psrc;
};

class im_warp
{
public:
    im_warp(
            img_t& in_src_img, 
            ctrl_points_t& in_src_ctrl_pts,
            img_t& in_dst_img,
            ctrl_points_t& in_dst_ctrl_pts)
        : 
    {
    }


private:
    warp_unit src;
    warp_unit dst;

    void setup_src()
    {
        // setup src control point
        // dump and plot
        // setup src tile_loc
        // dump and plot
    }

    void setup_dst()
    {
        // setup dst control point
        // dump and plot
        // setup dst tile_loc
        // dump and plot
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

    template <MappingFunctorType>
    static point_t<float> calc_mapping(const MappingFunctorType& mf, int dst_x, int dst_y)
    {
        return mf(dst_x, dst_y);
    }

    static pixel_t interpolate(
            const InterpFunctorType& interpf
            const float src_x,
            const float src_y)
    {
        return interpf(src_x, src_y);
    }

};

