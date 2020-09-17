#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <random>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <valarray>
#include <execution>
using namespace cv;
using namespace std;

const int NUM_IT = 500;

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

int kernel1(float ax, float ay){
    float x = 0.f; float y = 0.f;
    int n = 0;
    for(n = 0; n < NUM_IT ; ++n){
        float newx = x*x - y*y + ax;
        float newy = 2.f*x*y + ay;
        if(4.f < newx*newx + newy*newy) return n;
        x = newx; y = newy;
    }
    return NUM_IT;
}
int kernel2(float ax, float ay){
    float x = 0.f; float y = 0.f;
    int n = 0;
    for(int i = 0; i < NUM_IT; ++i) {
        const float newx = x*x - y*y + ax;
        const float newy = 2.f*x*y + ay;
        const int mask = 1 - ((int) (4.f < newx*newx + newy*newy)); 
        n += BITSELECT(mask, 1, 0);
        x =  BITSELECT(mask, newx, x);
        y =  BITSELECT(mask, newy, y);
    }
    return n;
}
template<class Kernel>
void mandelbrot_aos(std::vector<float>& arr, size_t X, size_t Y, Kernel kernel){
    const size_t XY = X*Y;
    for(size_t xy = 0; xy < XY; ++xy) {
        const float ax = (float) (xy % X) / (float) X;
        const float ay = (float) (xy / X) / (float) Y;
        arr[xy]        = (float) kernel(ax, ay); 
    }
}
inline void soa_kernel(size_t xy, 
                       std::vector<float>& arr,
                       std::vector<float>& xs, 
                       std::vector<float>& ys,
                       const std::vector<float>& axs,
                       const std::vector<float>& ays){
    const float newx = xs[xy] * xs[xy] - ys[xy] *ys[xy] + axs[xy];
    const float newy = 2.f * xs[xy]*ys[xy] + ays[xy];
    const int   mask = 1 - ((int) (4.f < newx*newx + newy*newy));
    arr[xy] += BITSELECT(mask, 1.f, 0.f);
    xs[xy]  = BITSELECT(mask, newx, xs[xy]);
    ys[xy]  = BITSELECT(mask, newy, ys[xy]);
}


void mandelbrot_soa(std::vector<float>& arr, size_t X, size_t Y){ 
    const size_t XY = X*Y;
    std::vector<float> xs(XY, 0.f), ys(XY, 0.f), axs(XY, 0.f), ays(XY, 0.f);
    for(size_t xy = 0; xy < XY; ++xy) {
        axs[xy] = ((float) (xy % X) / (float) X) / 200.f - 0.7463f;
        ays[xy] = ((float) (xy / X) / (float) Y) / 200.f + 0.1102f;
    }

    for(size_t i = 0; i < NUM_IT; ++i) {
        for(size_t xy = 0; xy < XY; ++xy) {
            soa_kernel(xy, arr,xs,ys,axs,ays);
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
#include <chrono>
using namespace chrono;
auto t1 = high_resolution_clock::now();
auto t2 = high_resolution_clock::now();
int main(int argc, char** argv){
   const int S = 1000;
   constexpr int XY = S*S;
   std::vector<float> arr1(XY), arr2(XY), arr3(XY);  
   t1 = high_resolution_clock::now(); 
   mandelbrot_aos(arr1, S, S, [] (float ax, float ay) { return kernel2(ax, ay);  } );
   t2 = high_resolution_clock::now();
   auto t_aos = duration_cast<duration<double>>(t2 - t1).count();

   t1 = high_resolution_clock::now();
   mandelbrot_soa(arr2, S, S);
   t2 = high_resolution_clock::now();
   auto t_soa = duration_cast<duration<double>>(t2 - t1).count();
   

   cout << "Time for AoS: " << t_aos << endl;
   cout << "Time for SoA: " << t_soa << endl;
   cout << "Improvement : " << (t_aos / t_soa) << " X" << endl;
   
   Mat img = toMat(arr2, S, S);  

   img /= NUM_IT;    
   resize(img, img, Size(1024, 768));
   imshow("Mandelbrot", img); 
   waitKey(0);
}
