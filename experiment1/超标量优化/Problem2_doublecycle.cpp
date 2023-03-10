#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace std;
#define n 33554432
unsigned long long int c[n];//n个数
unsigned long long int ans = 0;
int LOOP = 1;
void init()
{
    for(unsigned i =0;i<n;i++)
        c[i] = i;
}
int main()
{
    struct timeval starttime_3,endtime_3; 
    gettimeofday(&starttime_3,NULL);//begin
    for(int m = n;m>1;m/=2)
        for(int i = 0;i<m/2;i++)
            c[i] = c[i*2]+c[i*2+1];//相邻元素相加连续存储到数组最前面
    gettimeofday(&endtime_3,NULL);//end
    cout<<"doublecycle used time:"<<((endtime_3.tv_sec-starttime_3.tv_sec)*1000000+(endtime_3.tv_usec-starttime_3.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
    return 0;
}