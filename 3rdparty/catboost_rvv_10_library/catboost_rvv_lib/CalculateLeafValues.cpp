#include <riscv_vector.h>

#include "CalculateLeafValues.h"

#define CalculateLeafValues_rvv_def(MType)                                              \
void CalculateLeafValues_rvv_##MType(const size_t docCountInBlock,                      \
                         const double* treeLeafPtr,                                     \
                         const uint64_t* indexesPtr, double* writePtr)                  \
{                                                                                       \
  size_t cnt = docCountInBlock, bs;                                                     \
  vuint64##MType##_t ind_v;                                                             \
  vuint64##MType##_t ind_v_8;                                                           \
  vuint64##MType##_t val_u;                                                             \
  vfloat64##MType##_t val_treeLeaf;                                                     \
  vfloat64##MType##_t write;                                                            \
  vfloat64##MType##_t res;                                                              \
  bs = __riscv_vsetvl_e64##MType (cnt);                                                 \
  const auto docCountInBlock4 = (docCountInBlock | (bs - 1)) ^ (bs - 1);                \
  for (size_t docId = 0; docId < docCountInBlock4; docId += bs) {                       \
    ind_v   = __riscv_vle64_v_u64##MType (indexesPtr, bs);                              \
    ind_v_8 = __riscv_vsll_vx_u64##MType (ind_v, 3, bs);                                \
    val_treeLeaf = __riscv_vloxei64_v_f64##MType(treeLeafPtr, ind_v_8, bs);             \
    write = __riscv_vle64_v_f64##MType (writePtr, bs);                                  \
    res = __riscv_vfadd_vv_f64##MType (write, val_treeLeaf, bs);                        \
    __riscv_vse64_v_f64##MType (writePtr, res, bs);                                     \
    indexesPtr += bs;                                                                   \
    writePtr += bs;                                                                     \
  }                                                                                     \
  for (size_t docId = docCountInBlock4; docId < docCountInBlock; ++docId) {             \
      *writePtr += treeLeafPtr[*indexesPtr];                                            \
      ++writePtr;                                                                       \
      ++indexesPtr;                                                                     \
  }                                                                                     \
}

CalculateLeafValues_rvv_def(m4)
CalculateLeafValues_rvv_def(m8)

void CalculateLeafValues_rvv(const size_t docCountInBlock, const double* treeLeafPtr, 
                         const uint64_t* indexesPtr, double* writePtr)
{
  CalculateLeafValues_rvv_m4(docCountInBlock, treeLeafPtr, indexesPtr, writePtr);
}

#define LEAF_VALUES_BLOCK_STEP 4

void CalculateLeafValues_real(const size_t docCountInBlock, const double* treeLeafPtr, 
                         const uint64_t* indexesPtr, double* writePtr)
{
  const auto docCountInBlock4 = (docCountInBlock | 0x3) ^ 0x3;
  for (size_t docId = 0; docId < docCountInBlock4; docId += LEAF_VALUES_BLOCK_STEP) {
      writePtr[0] += treeLeafPtr[indexesPtr[0]];
      writePtr[1] += treeLeafPtr[indexesPtr[1]];
      writePtr[2] += treeLeafPtr[indexesPtr[2]];
      writePtr[3] += treeLeafPtr[indexesPtr[3]];
      writePtr += LEAF_VALUES_BLOCK_STEP;
      indexesPtr += LEAF_VALUES_BLOCK_STEP;
  }
  for (size_t docId = docCountInBlock4; docId < docCountInBlock; ++docId) {
      *writePtr += treeLeafPtr[*indexesPtr];
      ++writePtr;
      ++indexesPtr;
  }

}
