#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>

#include <riscv-vector.h>

#include "BinarizeFloats.h"

#define N 1000

void test()
{
  size_t docCount = 128;
  size_t cnt_border = 127;
  float borders[] = {2005.5, 2035.5, 2057.5, 2075.5, 2093.5, 2109.5, 2119.5, 2126.5, 2131.5, 2144.5, 2157.5, 2171.5, 2184.5, 2197.5, 2208.5, 2218.5, 2230.5, 2240.5, 2251.5, 2260.5, 2271.5, 2280.5, 2289.5, 2297.5, 2306.5, 2314.5, 2321.5, 2328.5, 2337.5, 2346.5, 2356.5, 2365.5, 2375.5, 2384.5, 2393.5, 2402.5, 2413.5, 2424.5, 2434.5, 2445.5, 2457.5, 2470.5, 2485.5, 2497.5, 2511.5, 2524.5, 2537.5, 2551.5, 2563.5, 2573.5, 2583.5, 2598.5, 2611.5, 2626.5, 2644.5, 2659.5, 2677.5, 2689.5, 2699.5, 2708.5, 2718.5, 2727.5, 2735.5, 2743.5, 2751.5, 2760.5, 2768.5, 2779.5, 2787.5, 2796.5, 2805.5, 2812.5, 2822.5, 2830.5, 2838.5, 2848.5, 2858.5, 2868.5, 2878.5, 2887.5, 2895.5, 2903.5, 2915.5, 2927.5, 2938.5, 2950.5, 2961.5, 2974.5, 2985.5, 3000.5, 3013.5, 3028.5, 3042.5, 3054.5, 3070.5, 3086.5, 3102.5, 3117.5, 3135.5, 3150.5, 3165.5, 3179.5, 3191.5, 3201.5, 3215.5, 3225.5, 3235.5, 3244.5, 3255.5, 3265.5, 3277.5, 3290.5, 3300.5, 3310.5, 3321.5, 3333.5, 3344.5, 3355.5, 3366.5, 3373.5, 3381.5, 3390.5, 3400.5, 3410.5, 3421.5, 3465.5, 3524}; 
  float val[] = {2680, 2683, 2713, 2709, 2706, 2699, 2699, 2696, 2696, 2693, 2693, 2686, 2690, 2686, 2680, 2670, 2722, 2722, 2722, 2722, 2719, 2716, 2709, 2709, 2703, 2703, 2699, 2699, 2696, 2690, 2693, 2686, 2680, 2670, 2660, 2726, 2732, 2729, 2732, 2729, 2732, 2732, 2732, 2732, 2729, 2722, 2719, 2713, 2713, 2709, 2706, 2703, 2696, 2690, 2680, 2673, 2660, 2650, 2647, 2808, 2808, 2811, 2732, 2736, 2736, 2739, 2736, 2739, 2739, 2739, 2736, 2739, 2742, 2742, 2742, 2742, 2736, 2732, 2722, 2719, 2719, 2713, 2706, 2696, 2683, 2673, 2663, 2650, 2640, 2644, 2808, 2804, 2801, 2801, 2804, 2739, 2736, 2742, 2745, 2745, 2749, 2745, 2749, 2745, 2749, 2745, 2745, 2752, 2749, 2752, 2749, 2745, 2739, 2732, 2729, 2726, 2719, 2709, 2693, 2676, 2663, 2650, 2640, 2631, 2634, 2818, 2811, 2804};
  size_t MAX_VALUES_PER_BIN = 254;
  uint8_t init_arr[]={0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  uint8_t res_arr[]={57, 57, 60, 60, 59, 58, 58, 58, 58, 58, 58, 57, 58, 57, 57, 56, 61, 61, 61, 61, 61, 60, 60, 60, 59, 59, 58, 58, 58, 58, 58, 57, 57, 56, 56, 61, 62, 62, 62, 62, 62, 62, 62, 62, 62, 61, 61, 60, 60, 60, 59, 59, 58, 58, 57, 56, 56, 55, 55, 71, 71, 71, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 62, 61, 61, 61, 60, 59, 58, 57, 56, 56, 55, 54, 54, 71, 70, 70, 70, 70, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 64, 65, 64, 64, 63, 62, 62, 61, 61, 60, 58, 56, 56, 55, 54, 54, 54, 72, 71, 70};

  BinarizeFloats_rvv(init_arr, val, docCount, docCount, borders, cnt_border, MAX_VALUES_PER_BIN);
  
  int f = 0;
  for(int i = 0; i < docCount; i++)
  {
    int diff = ((int)init_arr[i] - (int)res_arr[i]);
    if(diff != 0)
    {
      f = 1;
      std::cout << i << ":" << diff << " "<<(int)init_arr[i]<<" "<<(int)res_arr[i]<< std::endl; 
    }
  }
  if(f == 0)
  {
    std::cout << "Pass" << std::endl;
  }
  else
  {
    std::cout << "Fail" << std::endl;
  }
}


using namespace std;

vector<size_t>  a_docCount;
vector<size_t>  a_cnt_border;
vector<size_t>  a_MAX_VALUES_PER_BIN;
vector<vector<float> >   a_borders;
vector<vector<float> >   a_val;
vector<vector<uint8_t> > a_init_arr;
vector<vector<uint8_t> > a_res_arr;

int main(int argc, char **argv)
{
  int Ntest = 1;
  if(argc < 2)
  {
    printf("main <test> <N>");
    return -1;   
  }
  
  if(argc > 2)
  {
    Ntest = atoi(argv[2]);
  }
  printf("test cnt : %d\n", Ntest);
  
  test();
  
  ifstream finput(argv[1]);
  
  int max_diff_all_test = 0.0;
  int ind_diff_i = -1;

  for(int i = 0; i < Ntest; i++)
  {    
    uint32_t val; 
    size_t e_val_cnt;
    size_t e_cnt_border;
  
    finput >> e_val_cnt;
    finput >> e_cnt_border;
    cout<< "test     :"<< i  <<endl;
    cout<< "e_val_cnt   :"<< e_val_cnt   <<endl;
    cout<< "e_cnt_border:"<< e_cnt_border <<endl;
    
    vector<float> e_borders(e_cnt_border);
    vector<float> e_val   (e_val_cnt);
    vector<uint8_t> e_init(e_val_cnt);
    size_t e_MAX_VALUES_PER_BIN;
    vector<uint8_t> e_result(e_val_cnt);
          
    for(int j = 0; j < e_cnt_border; j++)
    {
      finput >> e_borders[j];
    }
    for(int j = 0; j < e_val_cnt; j++)
    {
      finput >> e_val[j];
    }
    finput >> e_MAX_VALUES_PER_BIN;
    
    for(int j = 0; j < e_val_cnt; j++)
    {
      finput >> val;
      e_init[j] = (uint8_t)val;
    }
    for(int j = 0; j < e_val_cnt; j++)
    {
      finput >> val;
      e_result[j] = (uint8_t)val;
    }
    
    a_docCount          .push_back(e_val_cnt);
    a_cnt_border        .push_back(e_cnt_border);
    a_MAX_VALUES_PER_BIN.push_back(e_MAX_VALUES_PER_BIN);
    a_borders .push_back(e_borders);
    a_val     .push_back(e_val);
    a_init_arr.push_back(e_init);
    a_res_arr .push_back(e_result);


    BinarizeFloats_rvv(e_init.data(), e_val.data(), e_val_cnt, e_val_cnt, e_borders.data(), e_cnt_border, e_MAX_VALUES_PER_BIN);

    int diff = 0;
    int max_diff = 0;
    int ind_diff = 0;
  
    int f = 0;
    for(int j = 0; j < e_val_cnt; j++)
    {
      diff = ((int)e_init[j] - (int)e_result[j]);
      if(diff != 0)
      {
        f = 1;
        std::cout << i << ":" << j <<"#" << diff << " "<<(int)e_init[j]<<" "<<(int)e_result[j]<< std::endl; 
      }
      if(max_diff < diff){
        max_diff = diff;
        ind_diff = j;
      }
    }
    
    if(max_diff_all_test < max_diff)
    {
        max_diff_all_test = max_diff;
        ind_diff_i = i;
    }
  }
  std::cout << "max diff :" << max_diff_all_test << std::endl;
  std::cout << "max diff ind:" << ind_diff_i << std::endl;
  
  
  std::chrono::steady_clock::time_point start ;
  std::chrono::steady_clock::time_point end;

  for(int j = 0; j < 10; j++)
  for(int i = 0; i < Ntest; i++)
  {
    BinarizeFloats_basic(
    a_init_arr[i].data(), 
    a_val[i].data(), 
    a_docCount[i], 
    a_docCount[i], 
    a_borders[i].data(), 
    a_cnt_border[i], 
    a_MAX_VALUES_PER_BIN[i]);
  }
  

  start = std::chrono::steady_clock::now();  
  
  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    BinarizeFloats_rvv(
    a_init_arr[i].data(), 
    a_val[i].data(), 
    a_docCount[i], 
    a_docCount[i], 
    a_borders[i].data(), 
    a_cnt_border[i], 
    a_MAX_VALUES_PER_BIN[i]);
  }
  end = std::chrono::steady_clock::now();
  auto diff = end - start; 
  double time = std::chrono::duration <double> (diff).count();
  std::cout << "BinarizeFloats_rvv(m4): " << time << std::endl;

  start = std::chrono::steady_clock::now();  
  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    BinarizeFloats_basic(
    a_init_arr[i].data(), 
    a_val[i].data(), 
    a_docCount[i], 
    a_docCount[i], 
    a_borders[i].data(), 
    a_cnt_border[i], 
    a_MAX_VALUES_PER_BIN[i]);
  }
  end = std::chrono::steady_clock::now();
  diff = end - start; 
  time = std::chrono::duration <double> (diff).count();
  std::cout << "BinarizeFloats_basic :" << time << std::endl;



}