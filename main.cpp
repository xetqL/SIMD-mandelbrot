#include <iostream>
#include <vector>

#include <x86intrin.h>    //AVX/SSE Extensions

using namespace std;

const int NUM_IT = 300;
const int S      = 4096;
constexpr int XY = S*S;
constexpr int N  = XY/8;

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
        const float ax = ((float) (xy % X) / (float) X) / 200.f - 0.7463f;
        const float ay = ((float) (xy / X) / (float) Y) / 200.f + 0.1102f;
        arr[xy]        = (float) kernel(ax, ay);
    }
}

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
            break;
        }

    }
    return res;
}

float* vals = (float*) aligned_alloc(32, 8);
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

#include <chrono>
using namespace chrono;
auto t1 = high_resolution_clock::now();
auto t2 = high_resolution_clock::now();

int main(int argc, char** argv){

   std::vector<float> arr1(XY), arr2(XY), arr3(XY);  
   
   t1 = high_resolution_clock::now(); 
   mandelbrot_aos_intr(arr1, S, S);
   t2 = high_resolution_clock::now();
   auto t_intr = duration_cast<duration<double>>(t2 - t1).count();

   t1 = high_resolution_clock::now();
   mandelbrot_aos(arr2, S, S, [] (float ax, float ay) {return kernel1(ax,ay);});
   t2 = high_resolution_clock::now();
   auto t_naiv = duration_cast<duration<double>>(t2 - t1).count();
   
   t1 = high_resolution_clock::now();
   mandelbrot_soa(arr3, S, S);
   t2 = high_resolution_clock::now();
   auto t_soa = duration_cast<duration<double>>(t2 - t1).count();
   
   cout << "Time intrinsic: " << t_intr << endl;
   cout << "Time naive    : " << t_naiv << endl;
   cout << "Time soa autov: " << t_soa << endl;
   cout << "Improvement   : " << (t_naiv / t_intr) << " X" << endl;

   return 0;
}
