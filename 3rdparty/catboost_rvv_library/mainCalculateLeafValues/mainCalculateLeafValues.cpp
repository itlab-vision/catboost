#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>

#include "CalculateLeafValues.h"

#define N 1000

using namespace std;

vector< vector<uint64_t>> a_ind;
vector< vector<double>> a_in_d;
vector< vector<double>> a_out_d;
vector< vector<double>> a_out_real;

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
    size_t cnt;
    finput >> cnt;

    vector<uint64_t> ind(cnt);    
    vector<double> in_d(cnt);    
    vector<double> out_d(cnt);
    vector<double> out_real(cnt);

    for(int j = 0; j < cnt; j++)
    {
      finput >> ind[j];
    }
    for(int j = 0; j < cnt; j++)
    {
      finput >> in_d[j];
    }
    for(int j = 0; j < cnt; j++)
    {
      finput >> out_d[j];
    }
    for(int j = 0; j < cnt; j++)
    {
      finput >> out_real[j];
    }

    a_ind.push_back(ind);
    a_in_d.push_back(in_d);
    a_out_d.push_back(out_d);
    a_out_real.push_back(out_real);
    
    CalculateLeafValues_rvv(cnt, in_d.data(), ind.data(), out_d.data());
    
    double diff = 0.0;
    double max_diff = 0.0;
    int ind_diff = 0;
    for(int j = 0; j < cnt; j++)
    {
      diff = fabs(out_d[j] - out_real[j]);
 
      if(diff > 0.001)
      printf("| %d %lf %lf %lf  |", j, diff, out_d[j], out_real[j]);
      
      if(max_diff < diff){
        max_diff = diff;
        ind_diff = j;
      }
    }
    if(max_diff_all_test < max_diff){
      max_diff_all_test = max_diff;
    }
    
    printf("\n    %e (%d  %.10lf %.10lf) \n", max_diff, ind_diff, out_d[ind_diff], out_real[ind_diff]); 
  }
  printf("max_diff_all_test: %e \n", max_diff_all_test);



  std::chrono::steady_clock::time_point start ;
  std::chrono::steady_clock::time_point end;

  for(int j = 0; j < 10; j++)
  for(int i = 0; i < Ntest; i++)
  {
    CalculateLeafValues_rvv(a_ind[i].size(), a_in_d[i].data(), a_ind[i].data(), a_out_d[i].data());
  }
  

  start = std::chrono::steady_clock::now();  
  
  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    CalculateLeafValues_rvv(a_ind[i].size(), a_in_d[i].data(), a_ind[i].data(), a_out_d[i].data());
  }
  end = std::chrono::steady_clock::now();
  auto diff = end - start; 
  double time = std::chrono::duration <double> (diff).count();
  std::cout << "CalculateLeafValues_rvv: " << time << std::endl;

  start = std::chrono::steady_clock::now();  
  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    CalculateLeafValues_real(a_ind[i].size(), a_in_d[i].data(), a_ind[i].data(), a_out_d[i].data());
  }
  end = std::chrono::steady_clock::now();
  diff = end - start; 
  time = std::chrono::duration <double> (diff).count();
  std::cout << "CalculateLeafValues_real: " << time << std::endl;

}




