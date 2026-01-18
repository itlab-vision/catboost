#pragma once

float L2SqrDistance_basic(const float* a, const float* b, int length);
float L2SqrDistance_rvv(const float* a, const float* b, int length);
float L2SqrDistance_rvv_m1(const float* a, const float* b, int length);
float L2SqrDistance_rvv_m2(const float* a, const float* b, int length);
float L2SqrDistance_rvv_m4(const float* a, const float* b, int length);
float L2SqrDistance_rvv_m8(const float* a, const float* b, int length);