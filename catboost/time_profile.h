#pragma once
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iomanip>


#define __USE_RVV___
#ifdef __USE_RVV___

void CalcIndexesBasic_without_xor_rvv(
            uint32_t* indexesVec,
            const uint8_t* binFeaturePtr,
            uint8_t borderVal,
            int depth,
            size_t start,
            size_t docCountInBlock);

void BinarizeFloats_rvv(uint8_t *writePtr, 
            const float *val_arr, 
            size_t val_cnt, 
            size_t docCount, 
            const float *borders, 
            size_t cnt_border, 
            int MAX_VALUES_PER_BIN);

float L2SqrDistance_rvv(const float* a, const float* b, int length);


#endif

#ifdef __TIME_PROF_2___
extern double CalcTreesBlockedImpl_time;
extern double CalculateLeafValuesMulti_time;
extern double CalcIndexesBasic_time;
extern double CalculateLeafValues_time;
extern double BinarizeFeatures_time;
extern double BinarizeFloatsNonSse_time;

extern double All_time;
#endif

#ifdef __TIME_PROF___

extern std::map<std::string, double> times;
extern std::map<std::string, int> times_count;
extern bool timing;
extern std::mutex m_timer;


class TimerForAlg
{
    std::chrono::steady_clock::time_point start ;
    std::chrono::steady_clock::time_point end;
    std::string s;
    bool print_f;
    public:
    TimerForAlg(std::string s, bool print_f=false)
    {
        std::unique_lock<decltype(m_timer)> lock(m_timer);
        this->s = s;
        this->print_f = print_f;
        if(print_f)
        {
            timing = true;            
        }
        start = std::chrono::steady_clock::now();
        times_count[s] += 1;
    }
    
    void print(){
        if(timing)
        {
            end = std::chrono::steady_clock::now();
            auto diff = end - start; 
            times[s]+=std::chrono::duration <double> (diff).count();
            start = std::chrono::steady_clock::now();
            times_count[s] += 1;
        }
        std::cout << "########## Time :" << std::endl;
        for (auto const &pair: times) {
            std::cout << "{ " << pair.first << ": " << pair.second <<": "<< times_count[pair.first] << std::endl;
        }
        timing = false;            
    }
    
    ~TimerForAlg()
    {
        std::unique_lock<decltype(m_timer)> lock(m_timer);
        if(print_f)
        {
            
        }
        else
        {
            if(timing)
            {
                end = std::chrono::steady_clock::now();
                auto diff = end - start; 
                times[s]+=std::chrono::duration <double> (diff).count();
            }
        }
    }
};

#endif