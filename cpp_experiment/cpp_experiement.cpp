#include "nlohmann/json.hpp"
#include <vector>
#include <iostream>
#include <fstream>

namespace Test2DVector
{
    template<typename T>
    struct v2d : std::vector<std::vector<T>>
    {
        v2d(int w, int h)
            : std::vector<std::vector<T>>(h, std::vector<T>(w))
        {
            for (int j=0; j<h; j++)
                for (int i=0; i<w; i++)
                    this->at(j).at(i) = i*j;
        }

        void print()
        {
            for (int j=0; j<this->size(); j++)
                for (int i=0; i<this->at(j).size(); i++)
                    std::cout << this->at(j).at(i) << ",";
        }
    };

    void Exe()
    {
        v2d<float> my_2dvec(5, 3);
        my_2dvec.print();
    }
}

namespace TestJsonArray
{
    void Exe()
    {
        std::ifstream ifs("C:\\HHWork\\ImWarping\\Data\\Input\\CppExperiment\\json_exp.json");
        //std::ifstream ifs("C:\\HHWork\\LW3D\\Development\\Data\\Input\\HtmAnalyzeFull\\INF_2020-02-04_11-27-17.bin");
        nlohmann::json js;

        ifs >> js;

        auto a = js["src"]["ctrl_points"]["content"];
        int ca[12];
        int idx = 0;
        for (auto i : a)
        {
            std::cout << i << std::endl;
            ca[idx] = i;
            idx++;
        }

        for (int i = 0; i < 12; i++)
            std::cout << ca[i] << std::endl;

    }
}

int main(int argc, char* argv[])
{
    Test2DVector::Exe();

    TestJsonArray::Exe();

    return 0;
}
