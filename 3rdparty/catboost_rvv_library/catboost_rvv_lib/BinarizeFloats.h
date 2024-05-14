#pragma once

#include <stdio.h>
#include <inttypes.h>


void BinarizeFloats_basic(uint8_t *writePtr, 
                          const float *val_arr, 
                          size_t val_cnt, 
                          size_t docCount, 
                          const float *borders, 
                          size_t cnt_border, 
                          int MAX_VALUES_PER_BIN);

void BinarizeFloats_rvv(uint8_t *writePtr, 
                          const float *val_arr, 
                          size_t val_cnt, 
                          size_t docCount, 
                          const float *borders, 
                          size_t cnt_border, 
                          int MAX_VALUES_PER_BIN);