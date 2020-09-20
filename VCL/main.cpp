#include <iostream>
#include <vector>
#include <array>
#include "vectorclass.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

const int NUM_IT = 500;
const int S      = 1024;
constexpr int XY = S*S;
constexpr int N  = XY/8;

Vec8f kernel_vcl(Vec8f ax, Vec8f ay){
    Vec8f x = 0.f, y = 0.f, count = 0.f;
    for(int n = 0; n < NUM_IT ; ++n){
        Vec8f newx = x*x - y*y + ax;
        Vec8f newy = 2.f * x*y + ay;
        Vec8fb mask= 4.f < newx*newx + newy*newy;
        count = select(mask, count, count + 1);
        x     = select(mask,     x,         newx);
        y     = select(mask,     y,         newy);
        if ( horizontal_and(mask) ) {
            return count;
        }
    }
    return count;
}
void mandelbrot_VCL(std::vector<float>& arr, size_t X, size_t Y){
    Vec8f ax8, ay8;
    for(int xy = 0; xy < XY; xy += 8) {
        for(int i = 0; i < 8; ++i) {
            ax8.insert(i, ((float) ((xy+i) % X) / (float) X ) / 200.f - 0.7463f);
            ay8.insert(i, ((float) ((xy+i) / X) / (float) Y ) / 200.f + 0.1102);
        }
        Vec8f count = kernel_vcl(ax8, ay8);
        count.store(arr.data() + xy);
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
    mandelbrot_VCL(arr1, S, S);
    t2 = high_resolution_clock::now();
    auto t_vcl = duration_cast<duration<double>>(t2 - t1).count();

    cout << "Time VCL    : " << t_vcl << endl;

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
