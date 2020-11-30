#include "spatial_f.h"

#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <numeric>

using namespace std;  

struct idx
{
    int idx1;
    int idx2;
};

class spatial  
{  
    public:  
        void func(float *points, idx &idx_c);   
};

void spatial::func(float *points, idx &idx_c)
{  
    vector<float> p1;
    vector<float> p2;
    vector<float> p3;   
    
    vector<float> mean(3);
    for(int i = 0; i < 1024; i++)
    {  
        p1.push_back(points[i * 3]);
        p2.push_back(points[i * 3 + 1]);
        p3.push_back(points[i * 3 + 2]);
        mean[0] += p1[i];
        mean[1] += p2[i];
        mean[2] += p3[i];
    }
    
	mean[0] = mean[0] / p1.size();    
	mean[1] = mean[1] / p2.size();  
	mean[2] = mean[2] / p3.size();  
    
    vector<float> d0;
    for(int i = 0; i < 1024; i++)
    {  
        float d = sqrt(pow(p1[i] - mean[0], 2) + pow(p2[i] - mean[1], 2) + pow(p3[i] - mean[2], 2));     
        d0.push_back(d);
    }
    vector<float>::iterator biggest = max_element(begin(d0), end(d0));
    int idx1 =  distance(begin(d0), biggest);

    for(int i = 0; i < 1024; i++)
    {  
        float d = sqrt(pow(p1[i] - p1[idx1], 2) + pow(p2[i] - p2[idx1], 2) + pow(p3[i] - p3[idx1], 2));     
        d0[i] += d;
    }
  
    int i = 0;
    int idx2 = 0;    
    while (1)
    {
        vector<float>::iterator biggest = max_element(begin(d0), end(d0));
        idx2 = distance(begin(d0), biggest);
        
        vector<float>vec1(3);
        vector<float>vec2(3);

        vec1[0] = p1[idx1] - mean[0];
        vec1[1] = p2[idx1] - mean[1];
        vec1[2] = p3[idx1] - mean[2];

        vec2[0] = p1[idx2] - mean[0];
        vec2[1] = p2[idx2] - mean[1];
        vec2[2] = p3[idx2] - mean[2];

        float angle = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
        float n1 = sqrt(pow(vec1[0], 2) + pow(vec1[1], 2) + pow(vec1[2], 2));   
        float n2 = sqrt(pow(vec2[0], 2) + pow(vec2[1], 2) + pow(vec2[2], 2));

        angle = angle / (n1 * n2);

        for(int i = 0; i < 1024; i++)
        {  
            float d = sqrt(pow(p1[i] - p1[idx2], 2) + pow(p2[i] - p2[idx2], 2) + pow(p3[i] - p3[idx2], 2));     
            d0[i] += d;
        }

        d0[idx1] = 0;
        d0[idx2] = 0;

        if (abs(angle) > 0.999)
        {
           if (i>=100)
           {
                cout<<"Its a line!!"<<endl;
                break;
           }
           else
           {
                i = i + 1;   
                continue;
           }
        }       
        else
            break; 
    }

    idx_c.idx1 = int(idx1);
    idx_c.idx2 = int(idx2);    
}  

extern "C" {  
    spatial spa; 
    void spa_f(float *points, idx &idx_c) 
    {  
        spa.func(points, idx_c);  
    //  return center;
    } 
    
}
