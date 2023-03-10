#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace std;
#define n 33554432
int sum[n];
int c[n];//n个数
int main()
{
    struct timeval starttime_1,endtime_1;
    float timeuse_1;
    //----------------超标量优化-----------------计算n个数的和
    for(int i=0;i<n;i++)
        c[i] = i%20;
    int ans = 0;
    //-----------------平凡算法-----------------
    gettimeofday(&starttime_1,NULL);//begin
    for(int i=0;i<n;i++)
        ans+=c[i];
    gettimeofday(&endtime_1,NULL);//end
    timeuse_1 = 1000000*(endtime_1.tv_sec-starttime_1.tv_sec)+(endtime_1.tv_usec-starttime_1.tv_usec);
    timeuse_1 /= 1000000;
    printf("timeuse_1 = %f\n",timeuse_1);
    cout<<ans<<endl;
    return 0;
}