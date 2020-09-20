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

inline int kernel(float ax, float ay){
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
void mandelbrot_aos(std::vector<float>& arr, size_t X, size_t Y){
    const size_t XY = X*Y;
    for(size_t xy = 0; xy < XY; ++xy) {
        const float ax = ((float) (xy % X) / (float) X) / 200.f - 0.7463f;
        const float ay = ((float) (xy / X) / (float) Y) / 200.f + 0.1102f;
        arr[xy]        = (float) kernel(ax, ay);
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
    mandelbrot_aos(arr1, S, S);
    t2 = high_resolution_clock::now();
    auto t_naiv = duration_cast<duration<double>>(t2 - t1).count();

    cout << "Time naive    : " << t_naiv << endl;

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
