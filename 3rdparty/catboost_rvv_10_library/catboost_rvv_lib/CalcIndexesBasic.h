#pragma once

#include <stdio.h>
#include <inttypes.h>

void CalcIndexesBasic_with_xor_basic(
      uint64_t* indexesVec,
			const uint64_t* binFeaturePtr,
      uint64_t xorMask,
			uint64_t borderVal,
			int depth,
			size_t start,
      size_t docCountInBlock);
			
			
void CalcIndexesBasic_with_xor_rvv(
      uint64_t* indexesVec,
			const uint64_t* binFeaturePtr,
      uint64_t xorMask,
			uint64_t borderVal,
			int depth,
			size_t start,
      size_t docCountInBlock);

void CalcIndexesBasic_without_xor_basic(
      uint32_t* indexesVec,
			const uint8_t* binFeaturePtr,
			uint8_t borderVal,
			int depth,
			size_t start,
      size_t docCountInBlock);
			
			
void CalcIndexesBasic_without_xor_rvv(
      uint32_t* indexesVec,
			const uint8_t* binFeaturePtr,
			uint8_t borderVal,
			int depth,
			size_t start,
      size_t docCountInBlock);

void CalcIndexesBasic_without_xor_rvv_m8(
      uint32_t* indexesVec,
			const uint8_t* binFeaturePtr,
			uint8_t borderVal,
			int depth,
			size_t start,
      size_t docCountInBlock);