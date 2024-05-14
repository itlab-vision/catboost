#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>

#include <riscv-vector.h>

#include "CalcIndexesBasic.h"

#define N 1000

using namespace std;

vector<size_t>   a_start;
vector<size_t>   a_finish;
vector<uint8_t>  a_border;
vector<uint32_t> a_depth;
vector<vector<uint8_t > > a_binF;
vector<vector<uint32_t> > a_ind;
vector<vector<uint32_t> > a_ind_res;

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
  
  ifstream finput(argv[1]);
  
  double max_diff_all_test = 0.0;

  
  for(int i = 0; i < Ntest; i++)
  {
    
    uint32_t val;
    size_t tmp;
    uint8_t tmp2;
    
    size_t e_start;
    size_t e_finish;
    uint8_t e_border;
    uint32_t e_depth;

    finput >> e_start;
    finput >> e_finish;
    finput >> val;
    e_border = (uint8_t) val;    
    finput >> e_depth;
    cout<< "test     :"<< i  <<endl;
    cout<< "e_start  :"<< e_start  <<endl;
    cout<< "e_finish :"<< e_finish <<endl;
    cout<< "e_border :"<< e_border <<endl;
    cout<< "e_depth  :"<< e_depth  <<endl;
    
    vector<uint8_t > e_binF(e_finish);
    vector<uint32_t> e_ind(e_finish);
    vector<uint32_t> e_test(e_finish);
    vector<uint32_t> e_test_m8(e_finish);
    vector<uint32_t> e_ind_res(e_finish);

    
    finput >> tmp;
    finput >> tmp;
    finput >> tmp;
        
    for(int j = 0; j < e_finish; j++)
    {
      finput >> val;
      e_binF[j] = (uint8_t)val;
    }
    for(int j = 0; j < e_finish; j++)
    {
      finput >> e_ind[j];
      e_test[j] = e_ind[j];
      e_test_m8[j] = e_ind[j];
    }
    for(int j = 0; j < e_finish; j++)
    {
      finput >> e_ind_res[j];
    }
    
    a_start  .push_back(e_start );
    a_finish .push_back(e_finish);
    a_border .push_back(e_border);
    a_depth  .push_back(e_depth );
    a_binF   .push_back(e_binF   );
    a_ind    .push_back(e_ind    );
    a_ind_res.push_back(e_ind_res);
    
    CalcIndexesBasic_without_xor_rvv(
      e_ind.data(),
			e_binF.data(),
			e_border,
			e_depth,
			e_start,
      e_finish);
    CalcIndexesBasic_without_xor_rvv(
      e_test_m8.data(),
			e_binF.data(),
			e_border,
			e_depth,
			e_start,
      e_finish);
    
    int diff = 0;
    int max_diff = 0.0;
    int ind_diff = 0;
    for(int j = 0; j < e_finish; j++)
    {
      diff = (e_ind[j] ^ e_ind_res[j])|(e_test_m8[j] ^ e_ind_res[j]);
 
      if(diff > 0)
      printf("| %d %lf %lf %lf  |", j, diff, e_ind[j], e_ind_res[j]);
      
      if(max_diff < diff){
        max_diff = diff;
        ind_diff = j;
      }
    }
    if(max_diff_all_test < max_diff){
      max_diff_all_test = max_diff;
    }
    
    printf("\n    %e (%d  %.10lf %.10lf) \n", max_diff, ind_diff, e_ind[ind_diff], e_ind_res[ind_diff]); 
  }
  printf("max_diff_all_test: %e \n", max_diff_all_test);
  
  
  
  std::chrono::steady_clock::time_point start ;
  std::chrono::steady_clock::time_point end;

  for(int j = 0; j < 10; j++)
  for(int i = 0; i < Ntest; i++)
  {
    CalcIndexesBasic_without_xor_rvv(
      a_ind [i].data(),
			a_binF[i].data(),
			a_border[i],
			a_depth [i],
			a_start [i],
      a_finish[i]);
  }
  

  start = std::chrono::steady_clock::now();  
  
  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    CalcIndexesBasic_without_xor_rvv(
      a_ind [i].data(),
			a_binF[i].data(),
			a_border[i],
			a_depth [i],
			a_start [i],
      a_finish[i]);
  }
  end = std::chrono::steady_clock::now();
  auto diff = end - start; 
  double time = std::chrono::duration <double> (diff).count();
  std::cout << "CalcIndexesBasic_without_xor_rvv(m4): " << time << std::endl;

  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    CalcIndexesBasic_without_xor_rvv_m8(
      a_ind [i].data(),
			a_binF[i].data(),
			a_border[i],
			a_depth [i],
			a_start [i],
      a_finish[i]);
  }
  end = std::chrono::steady_clock::now();
  diff = end - start; 
  time = std::chrono::duration <double> (diff).count();
  std::cout << "CalcIndexesBasic_without_xor_rvv(m8): " << time << std::endl;

  start = std::chrono::steady_clock::now();  
  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    CalcIndexesBasic_without_xor_basic(
      a_ind [i].data(),
			a_binF[i].data(),
			a_border[i],
			a_depth [i],
			a_start [i],
      a_finish[i]);
  }
  end = std::chrono::steady_clock::now();
  diff = end - start; 
  time = std::chrono::duration <double> (diff).count();
  std::cout << "CalcIndexesBasic_without_xor_basic: " << time << std::endl;

}
