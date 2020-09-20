#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <x86intrin.h>    //AVX/SSE Extensions

using namespace cv;
using namespace std;

const int NUM_IT = 300;
const int S      = 4096;
constexpr int XY = S*S;
constexpr int N  = XY/8;

float* vals = (float*) aligned_alloc(32, 8);
inline __m256 kernel(__m256 ax, __m256 ay)  {
    __m256 mone = _mm256_set1_ps(-1.0f);
    __m256 one  = _mm256_set1_ps(1.0f);
    __m256 two  = _mm256_set1_ps(2.0f);
    __m256 four = _mm256_set1_ps(4.0f);
    __m256 res  = _mm256_set1_ps(0.0f);
    __m256 x    = _mm256_set1_ps(0.0f);
    __m256 y    = _mm256_set1_ps(0.0f);

    for (int n = 0; n < NUM_IT; n++) {
        __m256 newx = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), ax);
        __m256 newy = _mm256_add_ps(_mm256_mul_ps(two, _mm256_mul_ps(x, y)), ay);
        __m256 norm = _mm256_add_ps(_mm256_mul_ps(newx, newx), _mm256_mul_ps(newy, newy));
        __m256 cmpmask = _mm256_cmp_ps(four, norm, _CMP_LT_OS);
        res = _mm256_blendv_ps(_mm256_add_ps(res, one), res, cmpmask);

        x = _mm256_blendv_ps(newx, x, cmpmask);
        y = _mm256_blendv_ps(newy, y, cmpmask);

        if(_mm256_testc_ps(cmpmask, mone) ){
            return res;
        }

    }
    return res;
}
void mandelbrot_aos_intr(std::vector<float>& arr, size_t X, size_t Y){
    const size_t XY = X*Y;
    __m256 ax = _mm256_set1_ps(-0.7463f);
    __m256 ay = _mm256_set1_ps( 0.1102f);
    for(size_t xy = 0; xy < XY; xy +=8) {
        for(int i = 0; i < 8; ++i) {
            ax[i] += ((float) ((xy+i) % X) / (float) X) / 200.f;
            ay[i] += ((float) ((xy+i) / X) / (float) Y) / 200.f;
        }

        __m256 res = kernel(ax, ay);
        _mm256_store_ps(vals,  res);

        std::copy(vals, vals+8, arr.begin() + xy); 

        ax = _mm256_set1_ps(-0.7463f);
        ay = _mm256_set1_ps( 0.1102f);
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
    mandelbrot_aos_intr(arr1, S, S);
    t2 = high_resolution_clock::now();
    auto t_intr = duration_cast<duration<double>>(t2 - t1).count();

    cout << "Time: " << t_intr << endl;

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
