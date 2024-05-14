#include <riscv-vector.h>
#include "L2SqrDistance.h"
 
inline float SqrDelta_basic(float a, float b) {
  float diff = a < b ? b - a : a - b;
  return diff * diff;
}

float L2SqrDistance_basic(const float* a, const float* b, int length) {
  float res = 0;

  for (int i = 0; i < length; i++) {
    res += SqrDelta_basic(a[i], b[i]);
  }

  return res;
}


#define L2SqrDistance_rvv_def(MType)                                   \
float L2SqrDistance_rvv_##MType(const float* a, const float* b, int length) {  \
  size_t bs;                                                           \
  size_t bs2;                                                          \
                                                                       \
  vfloat32##MType##_t v_a, v_b, v_diff, v_mul, v_sum;                  \
                                                                       \
  bs2 = vsetvl_e32m1(length);                                          \
  bs = vsetvl_e32##MType(length);                                      \
  v_sum = vfmv_v_f_f32##MType(0, bs);                                  \
  vfloat32m1_t v_res = vfmv_v_f_f32m1(0, bs2);                         \
  float res;                                                           \
                                                                       \
  bs = vsetvl_e32##MType(length);                                      \
  for (; length > bs; length -= bs) {                                  \
    v_a = vle_v_f32##MType(a, bs);                                     \
    v_b = vle_v_f32##MType(b, bs);                                     \
    v_diff = vfsub_vv_f32##MType (v_a, v_b, bs);                       \
    v_sum = vfmacc_vv_f32##MType (v_sum, v_diff, v_diff, bs);          \
    a += bs;                                                           \
    b += bs;                                                           \
                                                                       \
  }                                                                    \
  v_res = vfredsum_vs_f32##MType##_f32m1(v_res, v_sum, v_res, bs);     \
  if(length > 0)                                                       \
  {                                                                    \
    bs = vsetvl_e32##MType(length);                                    \
    v_a = vle_v_f32##MType(a, bs);                                     \
    v_b = vle_v_f32##MType(b, bs);                                     \
    v_diff = vfsub_vv_f32##MType(v_a, v_b, bs);                        \
    v_mul  = vfmul_vv_f32##MType(v_diff, v_diff, bs);                  \
    v_res = vfredsum_vs_f32##MType##_f32m1(v_res, v_mul, v_res, bs);   \
  }                                                                    \
  vse_v_f32m1(&res, v_res, 1);                                         \
  return res;                                                          \
}

L2SqrDistance_rvv_def(m1)
L2SqrDistance_rvv_def(m2)
L2SqrDistance_rvv_def(m4)
L2SqrDistance_rvv_def(m8)

float L2SqrDistance_rvv(const float* a, const float* b, int length)
{
  return L2SqrDistance_rvv_m4(a, b, length);
}