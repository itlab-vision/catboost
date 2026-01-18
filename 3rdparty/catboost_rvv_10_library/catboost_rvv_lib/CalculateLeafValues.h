#pragma once
#include <stdio.h>
#include <inttypes.h>

void CalculateLeafValues_rvv(const size_t docCountInBlock, const double* treeLeafPtr, 
                         const uint64_t* indexesPtr, double* writePtr);

void CalculateLeafValues_real(const size_t docCountInBlock, const double* treeLeafPtr, 
                         const uint64_t* indexesPtr, double* writePtr);