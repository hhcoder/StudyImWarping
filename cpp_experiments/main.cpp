#include <vector>
#include <iostream>

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

int main(int argc, char* argv[])
{
    Test2DVector::Exe();

    return 0;
}
