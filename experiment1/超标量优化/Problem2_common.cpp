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
    init();
    struct timeval starttime_1,endtime_1;
    //-----------------平凡算法-----------------
    gettimeofday(&starttime_1,NULL);//begin

    for(int k = 1;k<=LOOP;k++)
    {
        for(int i=0;i<n;i++)  ans+=c[i];
    }
    gettimeofday(&endtime_1,NULL);//end
    cout<<"common used time:"<<((endtime_1.tv_sec-starttime_1.tv_sec)*1000000+(endtime_1.tv_usec-starttime_1.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
    return 0;
}