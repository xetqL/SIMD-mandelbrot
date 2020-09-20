#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

const int NUM_IT = 300;
const int S      = 4096;
constexpr int XY = S*S;

inline int BITSELECT(int condition, int truereturnvalue, int falsereturnvalue){
    return (truereturnvalue & -condition) | (falsereturnvalue & ~(-condition)); 
}
inline float BITSELECT(int condition, float truereturnvalue, float falsereturnvalue)
{
    int& at = reinterpret_cast<int&>(truereturnvalue);
    int& af = reinterpret_cast<int&>(falsereturnvalue);
    int res = (at & -condition) | (af & ~(-condition)); //a when TRUE and b when FALSE
    return  reinterpret_cast<float&>(res);
}

void mandelbrot_soa(std::vector<float>& arr, size_t X, size_t Y){ 
    const size_t XY = X*Y;
    std::vector<float> xs(XY, 0.f), ys(XY, 0.f), axs(XY, 0.f), ays(XY, 0.f);
    for(size_t xy = 0; xy < XY; ++xy) {
        axs[xy] = ((float) (xy % X) / (float) X) / 200.f - 0.7463f;
        ays[xy] = ((float) (xy / X) / (float) Y) / 200.f + 0.1102f;
    }

    for(size_t i = 0; i < NUM_IT; ++i) {
        #pragma GCC ivdep
        for(size_t xy = 0; xy < XY; ++xy) {
          const float newx = xs[xy] * xs[xy] - ys[xy] *ys[xy] + axs[xy];
          const float newy = 2.f * xs[xy]*ys[xy] + ays[xy];
          const int   mask = 1 - ((int) (4.f < newx*newx + newy*newy));
          arr[xy] += BITSELECT(mask, 1.f, 0.f);
           xs[xy]  = BITSELECT(mask, newx, xs[xy]);
           ys[xy]  = BITSELECT(mask, newy, ys[xy]);
        }
    }
}
Mat toMat(const std::vector<float>& arr, size_t X, size_t Y){
    Mat img(X,Y, CV_32F);

    for(int i=0; i<X; ++i)
        for(int j=0; j<Y; ++j)
            img.at<float>(i, j) = arr.at(j + i * X);

    return img;
}

void print(Mat mat, size_t X, size_t Y){
    for(int i=0; i<X; ++i) {
        for (int j = 0; j < Y; ++j) {
            cout << mat.at<float>(i, j) << " ";
        }
        cout << endl;
    }
}
#include <chrono>
using namespace chrono;
auto t1 = high_resolution_clock::now();
auto t2 = high_resolution_clock::now();

void runMandelbrot(){
    std::vector<float> arr1(XY);

    t1 = high_resolution_clock::now();
    mandelbrot_soa(arr1, S, S);
    t2 = high_resolution_clock::now();
    auto t_soa = duration_cast<duration<double>>(t2 - t1).count();

    cout << "Time soa autov: " << t_soa << endl;

    Mat img = toMat(arr1, S, S);

    img /= NUM_IT;

    resize(img, img, Size(1024, 768));
    imshow("Mandelbrot", img);
    waitKey(0);
}

int main(int argc, char** argv){
   runMandelbrot();
   return 0;
}
