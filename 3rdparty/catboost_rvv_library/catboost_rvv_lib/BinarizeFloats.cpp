#include <riscv-vector.h>

#include "BinarizeFloats.h"

size_t Min(size_t a, size_t b)
{
  return a < b ? a : b;
}

void BinarizeFloats_basic(uint8_t *result, 
                          const float *val_arr, 
                          size_t val_cnt, 
                          size_t docCount, 
                          const float *borders, 
                          size_t cnt_border, 
                          int MAX_VALUES_PER_BIN)
{
  for (size_t docId = 0; docId < val_cnt; ++docId) {
    float val = val_arr[docId];
    
    uint8_t* writePtr = result + docId;
    for (size_t blockStart = 0; blockStart < cnt_border; blockStart += MAX_VALUES_PER_BIN) {
        const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, cnt_border);
        for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
            *writePtr += (uint8_t)(val > borders[borderId]);
        }
        writePtr += docCount;
    }
  }

}

void BinarizeFloats_rvv(uint8_t *result, 
                        const float *val_arr, 
                        size_t val_cnt, 
                        size_t docCount, 
                        const float *borders, 
                        size_t cnt_border, 
                        int MAX_VALUES_PER_BIN)
{
  size_t bs, m_bs, c_bs;
  bs = vsetvl_e32m4(val_cnt);
  c_bs = val_cnt / bs;
  m_bs = val_cnt % bs;

  vuint8m1_t one_v = vmv_v_x_u8m1(1, bs);
  
  vfloat32m4_t val_v;
  vuint8m1_t writePtr_v;
  
  for (size_t docId = 0; docId < val_cnt; docId += bs) {
    val_v = vle_v_f32m4(val_arr + docId, bs);
    
    uint8_t* writePtr = result + docId;
    writePtr_v = vle_v_u8m1(writePtr, bs); 

    for (size_t blockStart = 0; blockStart < cnt_border; blockStart += MAX_VALUES_PER_BIN) {
        const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, cnt_border);
        for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
          vbool8_t val_gt_border_m = vmfgt_vf_f32m4_b8 (val_v, borders[borderId], bs);
          writePtr_v = vadd_vv_u8m1_m (val_gt_border_m, writePtr_v, writePtr_v, one_v, bs);
        }
        vse_v_u8m1(writePtr, writePtr_v, bs);
        writePtr += docCount;
    }
  }  
  
  if(m_bs > 0)
  {
    size_t docId = m_bs * c_bs;
    bs = m_bs;
    val_v = vle_v_f32m4(val_arr + docId, bs);
    
    uint8_t* writePtr = result + docId;
    writePtr_v = vle_v_u8m1(writePtr, bs); 

    for (size_t blockStart = 0; blockStart < cnt_border; blockStart += MAX_VALUES_PER_BIN) {
        const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, cnt_border);
        for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
          vbool8_t val_gt_border_m = vmfgt_vf_f32m4_b8 (val_v, borders[borderId], bs);
          writePtr_v = vadd_vv_u8m1_m (val_gt_border_m, writePtr_v, writePtr_v, one_v, bs);
        }
        vse_v_u8m1(writePtr, writePtr_v, bs);
        writePtr += docCount;
    }
  }
  
}
