#include <riscv-vector.h>

#include "CalcIndexesBasic.h"

void CalcIndexesBasic_with_xor_rvv(
            uint64_t* indexesVec,
            const uint64_t* binFeaturePtr,
            uint64_t xorMask,
            uint64_t borderVal,
            int depth,
            size_t start,
            size_t docCountInBlock)
{
  size_t cnt = docCountInBlock, bs;
  bs = vsetvl_e64m8(cnt);
    
  vuint64m8_t indexesVec_v;
  vuint64m8_t binFeature_v;
  vuint64m8_t binFeature_xor_v;
  vbool8_t    binFeature_xor_ge_v;
  vuint64m8_t one_v = vmv_v_x_u64m8(1, bs);

  vuint64m8_t one_depth_v = vsll_vx_u64m8(one_v, depth, bs);

  
  const auto docCountInBlock4 = (docCountInBlock | (bs - 1)) ^ (bs - 1);
  for (size_t docId = 0; docId < docCountInBlock4; docId += bs) {
    indexesVec_v = vle_v_u64m8 (indexesVec+docId, bs);
    binFeature_v = vle_v_u64m8 (binFeaturePtr+docId, bs);
    binFeature_xor_v = vxor_vx_u64m8(binFeature_v, xorMask, bs);
    
    binFeature_xor_ge_v = vmsgeu_vx_u64m8_b8 (binFeature_xor_v, borderVal, bs);

    indexesVec_v = vor_vv_u64m8_m (binFeature_xor_ge_v, indexesVec_v, indexesVec_v, one_depth_v, bs);
    vse_v_u64m8 (indexesVec+docId, indexesVec_v, bs);
  }

  for (size_t docId = docCountInBlock4; docId < docCountInBlock; ++docId) {
    indexesVec[docId] |= ((binFeaturePtr[docId] ^ xorMask) >= borderVal) << depth;
  }
}


void CalcIndexesBasic_without_xor_basic(
      uint32_t* indexesVec,
      const uint8_t* binFeaturePtr,
      uint8_t borderVal,
      int depth,
      size_t start,
      size_t docCountInBlock)
{
  for (size_t docId = start; docId < docCountInBlock; ++docId) {
    indexesVec[docId] |= ((binFeaturePtr[docId]) >= borderVal) << depth;
  } 
}
      
      
void CalcIndexesBasic_without_xor_rvv(
      uint32_t* indexesVec,
      const uint8_t* binFeaturePtr,
      uint8_t borderVal,
      int depth,
      size_t start,
      size_t docCountInBlock)
{
  size_t cnt = docCountInBlock, bs, m_bs, c_bs;
  bs = vsetvl_e32m4(cnt);
  c_bs = cnt / bs;
  m_bs = cnt % bs;

  vuint32m4_t indexesVec_v;
  vuint8m1_t  binFeature_v;
  vbool8_t    binFeature_ge_v;
  
  vuint32m4_t one_v = vmv_v_x_u32m4(1, bs);
  vuint32m4_t one_depth_v = vsll_vx_u32m4(one_v, depth, bs);

  
  const auto docCountInBlock4 = c_bs * bs;
  for (size_t docId = 0; docId < docCountInBlock4; docId += bs) {
    indexesVec_v = vle_v_u32m4 (indexesVec+docId, bs);
    binFeature_v = vle_v_u8m1 (binFeaturePtr+docId, bs);
    
    binFeature_ge_v = vmsgeu_vx_u8m1_b8 (binFeature_v, borderVal, bs);

    indexesVec_v = vor_vv_u32m4_m (binFeature_ge_v, indexesVec_v, indexesVec_v, one_depth_v, bs);
    vse_v_u32m4 (indexesVec+docId, indexesVec_v, bs);
  }

  if(m_bs>0)
  {
    size_t docId = docCountInBlock4;
    bs = m_bs;
    indexesVec_v = vle_v_u32m4 (indexesVec+docId, bs);
    binFeature_v = vle_v_u8m1 (binFeaturePtr+docId, bs);
    
    binFeature_ge_v = vmsgeu_vx_u8m1_b8 (binFeature_v, borderVal, bs);

    indexesVec_v = vor_vv_u32m4_m (binFeature_ge_v, indexesVec_v, indexesVec_v, one_depth_v, bs);
    vse_v_u32m4 (indexesVec+docId, indexesVec_v, bs);
  }
  
}

void CalcIndexesBasic_without_xor_rvv_m8(
      uint32_t* indexesVec,
      const uint8_t* binFeaturePtr,
      uint8_t borderVal,
      int depth,
      size_t start,
      size_t docCountInBlock)
{
  size_t cnt = docCountInBlock, bs, m_bs, c_bs;
  bs = vsetvl_e32m8(cnt);
  c_bs = cnt / bs;
  m_bs = cnt % bs;

  vuint32m8_t indexesVec_v;
  vuint8m2_t  binFeature_v;
  vbool4_t    binFeature_ge_v;
  vuint32m8_t one_v = vmv_v_x_u32m8(1, bs);
  vuint32m8_t one_depth_v = vsll_vx_u32m8(one_v, depth, bs);

  const auto docCountInBlock4 = c_bs * bs;
  for (size_t docId = 0; docId < docCountInBlock4; docId += bs) {
    indexesVec_v = vle_v_u32m8 (indexesVec+docId, bs);
    binFeature_v = vle_v_u8m2 (binFeaturePtr+docId, bs);
    
    binFeature_ge_v = vmsgeu_vx_u8m2_b4 (binFeature_v, borderVal, bs);

    indexesVec_v = vor_vv_u32m8_m (binFeature_ge_v, indexesVec_v, indexesVec_v, one_depth_v, bs);
    vse_v_u32m8 (indexesVec+docId, indexesVec_v, bs);
  }

  if(m_bs>0)
  {
    size_t docId = docCountInBlock4;
    bs = m_bs;
    indexesVec_v = vle_v_u32m8 (indexesVec+docId, bs);
    binFeature_v = vle_v_u8m2 (binFeaturePtr+docId, bs);
    
    binFeature_ge_v = vmsgeu_vx_u8m2_b4 (binFeature_v, borderVal, bs);

    indexesVec_v = vor_vv_u32m8_m (binFeature_ge_v, indexesVec_v, indexesVec_v, one_depth_v, bs);
    vse_v_u32m8 (indexesVec+docId, indexesVec_v, bs);
  }  
}

