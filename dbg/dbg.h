#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <tuple>
#include <array>
#include <iterator>
#include <utility>
#include <sstream>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include <Windows.h>
#include <direct.h>
#endif

#include "../external/json/single_include/nlohmann/json.hpp"

namespace dbg
{
    const std::string fs_create_dir(const std::string& in_dir)
    {
        if (0 == in_dir.size())
        {
            return { "global output dir is not set!" };
        }
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
        // Recursively create directory - thx, StackOverflow
        std::size_t pos = 0;
        do
        {
            pos = in_dir.find_first_of("/", pos + 1);
            // support ASCII version only
            if (CreateDirectoryA(in_dir.substr(0, pos).c_str(), NULL) != 0 &&
                ERROR_ALREADY_EXISTS != GetLastError())
            {
                return { "Unknown error creating directory" };
            }
        } while (pos != std::string::npos);
#endif
        return in_dir;
    }

    const std::string format_num(int num)
    {
        std::stringstream ss;
        ss << std::setw(3) << std::setfill('0') << num;
        return ss.str();
    }

    std::string replace_char(const std::string& in, const char orig, const char to_replace)
    {
        std::string ret = in;
        for (std::size_t i = 0; i < in.length(); i++)
        {
            if (ret[i] == orig) ret[i] = to_replace;
        }
        return ret;
    }

    std::string validate_file_name(const std::string& in)
    {
        std::string ret = replace_char(in, ' ', '_');
        //ret = replace_char(ret, '-', '_');
        //ret = replace_char(ret, '[', '_');
        //ret = replace_char(ret, ']', '_');
        //ret = replace_char(ret, ')', '_');
        //ret = replace_char(ret, '(', '_');
        return ret;
    }

    template <typename T>
    static const std::string write_img(
        const std::string& in_format, 
        const T* buf, 
        const std::size_t in_size,
        const std::string bin_path)
    {

        using namespace std::string_literals;

        std::ofstream ofs(bin_path, std::ios::out | std::ios::binary);
        if (!ofs)
        {
            return { "error open file: "s + bin_path };
        }
        else
        {
            std::size_t buf_size = sizeof(T) * in_size;
            ofs.write((char*)buf, buf_size);
            return bin_path;
        }
    }
    using dim_t = std::array<std::size_t, 4>;

    struct jdump : public nlohmann::json
    {
        jdump(const std::string& in_dir, const std::string& in_fname, const std::string& in_fext)
            : out_dir(fs_create_dir(in_dir)),
              fname(validate_file_name(in_fname)),
              fext(in_fext),
              image_file_counter(0)
        { }

        template <typename ...Args>
        decltype(auto) operator[](Args&& ... args)
        {
            return nlohmann::json::operator[](std::forward<Args>(args)...);
        }

        template <typename ...Args>
        decltype(auto) operator=(Args&& ... args)
        {
            return nlohmann::json::operator=(std::forward<Args>(args)...);
        }

        ~jdump()
        {
            std::cout << std::setw(4) << *this << std::endl;

            std::ofstream ofs(out_dir + fname + fext, std::ios::out);
            ofs << std::setw(4) << *this << std::endl;
        }

        template <typename T>
        decltype(auto) c_array(const T* ptr, std::size_t len)
        {
            return std::vector<T>(&(ptr[0]), &(ptr[len - 1]));
        }

        static std::string str_tram_endswith(const std::string& in, const std::string& to_find)
        {
            return std::string(&in[0], &in[in.rfind(to_find)]);
        }

        template <typename T>
        decltype(auto) img(
            const std::string& in_name,
            const std::string& in_format,
            const T* buf,
            const dim_t& in_dim)
        {
            using namespace std::string_literals;
            const std::string bin_path =
                out_dir +
                fname +
                "_img_"s +
                in_name +
                ".bin"s;

            nlohmann::json j;

            j["format"] = in_format;
            j["width"] = in_dim[0];
            j["height"] = in_dim[1];
            j["stride"] = in_dim[2];
            j["channels"] = in_dim[3];
            j["file_loc"] = write_img(in_format, buf, 
                (size_t)in_dim[1] * (size_t)in_dim[2] * (size_t)in_dim[3], bin_path);

            return j;
        }

        std::string out_dir;
        std::string fname;
        std::string fext;
        int image_file_counter;
    };
}

//#define DBG_DUMP__ADAPT_STRUCT(_name_, _v0_, _v1_, _v2_, _v3_)        ;

// check out "pretty print" on Github: http://louisdx.github.io/cxx-prettyprint/
//struct print_vars {
//    template <typename T>
//    void operator()(T const& x) const {
//        std::cout << '<' << typeid(x).name() << ' ' << x << '>';
//    };
//};

