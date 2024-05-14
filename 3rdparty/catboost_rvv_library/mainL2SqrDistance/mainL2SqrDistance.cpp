#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>

#include <riscv-vector.h>

#include "L2SqrDistance.h"
 
#define N 1000
#define TEST_VECTOR_LEN 31

using namespace std;

void test()
{
  int len = TEST_VECTOR_LEN;
  float input_a[TEST_VECTOR_LEN] = {
    -0.4325648115282207, -1.6655843782380970,
    0.1253323064748307, 0.2876764203585489, -1.1464713506814637,
    1.1909154656429988, 1.1891642016521031, -0.0376332765933176,
    0.3272923614086541, 0.1746391428209245, -0.1867085776814394,
    0.7257905482933027, -0.5883165430141887, 2.1831858181971011,
    -0.1363958830865957, 0.1139313135208096, 1.0667682113591888,
    0.0592814605236053, -0.0956484054836690, -0.8323494636500225,
    0.2944108163926404, -1.3361818579378040, 0.7143245518189522,
    1.6235620644462707, -0.6917757017022868, 0.8579966728282626,
    1.2540014216025324, -1.5937295764474768, -1.4409644319010200,
    0.5711476236581780, -0.3998855777153632};
  float input_b[TEST_VECTOR_LEN] = {
    1.7491401329284098, 0.1325982188803279, 0.3252281811989881,
    -0.7938091410349637, 0.3149236145048914,-0.5272704888029532,
    0.9322666565031119, 1.1646643544607362,-2.0456694357357357,
    -0.6443728590041911, 1.7410657940825480, 0.4867684246821860,
    1.0488288293660140, 1.4885752747099299, 1.2705014969484090,
    -1.8561241921210170, 2.1343209047321410, 1.4358467535865909,
    -0.9173023332875400, -1.1060770780029008, 0.8105708062681296,
    0.6985430696369063, -0.4015827425012831, 1.2687512030669628,
    -0.7836083053674872, 0.2132664971465569, 0.7878984786088954,
    0.8966819356782295, -0.1869172943544062, 1.0131816724341454,
    0.2484350696132857};
 
  float res_basic = L2SqrDistance_basic(input_a, input_b, len);
  float res_rvv   = L2SqrDistance_rvv  (input_a, input_b, len);
  float res_rvv_m8= L2SqrDistance_rvv_m8(input_a, input_b, len);
 
  cout << res_basic << endl;
  cout << res_rvv << endl;
  cout << res_rvv_m8 << endl;
}

vector<vector<float> > a_vec_arr;
vector<vector<float> > b_vec_arr;

int main(int argc, char **argv)
{
  int Ntest = 1;
  if(argc < 2)
  {
    printf("main <N>");
    return -1;   
  }
  
  if(argc > 1)
  {
    Ntest = atoi(argv[1]);
  }
  printf("test cnt : %d\n", Ntest);
  
  test();
  
  
  srand(1111);

  int vec_size = 128;  
  for(int i = 0; i < Ntest; i++)
  {
    vector<float> a_val(vec_size);
    vector<float> b_val(vec_size);
    for(int j = 0; j < vec_size; j++)
    {
      a_val[j] = (float)rand()/RAND_MAX;
      b_val[j] = (float)rand()/RAND_MAX;
    }
    a_vec_arr.push_back(a_val);
    b_vec_arr.push_back(b_val);
  }
  
  
  std::chrono::steady_clock::time_point start ;
  std::chrono::steady_clock::time_point end;

  for(int j = 0; j < 10; j++)
  for(int i = 0; i < Ntest; i++)
  {
    L2SqrDistance_basic(a_vec_arr[i].data(), b_vec_arr[i].data(), vec_size);
  }
  

  start = std::chrono::steady_clock::now();  
  
  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    L2SqrDistance_rvv(a_vec_arr[i].data(), b_vec_arr[i].data(), vec_size);
  }
  end = std::chrono::steady_clock::now();
  auto diff = end - start; 
  double time1 = std::chrono::duration <double> (diff).count();
  std::cout << "L2SqrDistance_rvv: " << time1 << std::endl;


  start = std::chrono::steady_clock::now();  
  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    L2SqrDistance_rvv_m1(a_vec_arr[i].data(), b_vec_arr[i].data(), vec_size);
  }
  end = std::chrono::steady_clock::now();
  diff = end - start; 
  double time1_m1 = std::chrono::duration <double> (diff).count();
  std::cout << "L2SqrDistance_rvv(m1): " << time1_m1 << std::endl;

  start = std::chrono::steady_clock::now();  
  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    L2SqrDistance_rvv_m2(a_vec_arr[i].data(), b_vec_arr[i].data(), vec_size);
  }
  end = std::chrono::steady_clock::now();
  diff = end - start; 
  double time1_m2 = std::chrono::duration <double> (diff).count();
  std::cout << "L2SqrDistance_rvv(m2): " << time1_m2 << std::endl;

  start = std::chrono::steady_clock::now();  
  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    L2SqrDistance_rvv_m4(a_vec_arr[i].data(), b_vec_arr[i].data(), vec_size);
  }
  end = std::chrono::steady_clock::now();
  diff = end - start; 
  double time1_m4 = std::chrono::duration <double> (diff).count();
  std::cout << "L2SqrDistance_rvv(m4): " << time1_m4 << std::endl;




  start = std::chrono::steady_clock::now();  
  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    L2SqrDistance_rvv_m8(a_vec_arr[i].data(), b_vec_arr[i].data(), vec_size);
  }
  end = std::chrono::steady_clock::now();
  diff = end - start; 
  double time1_m8 = std::chrono::duration <double> (diff).count();
  std::cout << "L2SqrDistance_rvv(m8): " << time1_m8 << std::endl;

  start = std::chrono::steady_clock::now();  
  for(int j = 0; j < 100; j++)
  for(int i = 0; i < Ntest; i++)
  {
    L2SqrDistance_basic(a_vec_arr[i].data(), b_vec_arr[i].data(), vec_size);
  }
  end = std::chrono::steady_clock::now();
  diff = end - start; 
  double time2 = std::chrono::duration <double> (diff).count();
  std::cout << "L2SqrDistance_basic :" << time2 << std::endl;
  std::cout << "speedup(m1) :" << time2 / time1_m1 << std::endl;
  std::cout << "speedup(m2) :" << time2 / time1_m2 << std::endl;
  std::cout << "speedup(m4) :" << time2 / time1_m4 << std::endl;
  std::cout << "speedup(m8) :" << time2 / time1_m8 << std::endl;



}