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
    struct timeval starttime_2,endtime_2;
    float timeuse_2;
    //----------------超标量优化-----------------计算n个数的和
    for(int i=0;i<n;i++)
        c[i] = i%20;
    int ans = 0;

    //-----------------优化算法------------------
    //多链路式
    ans = 0;
    int sum1 = 0;
    int sum2 = 0;
    gettimeofday(&starttime_2,NULL);//begin
    for(int i=0;i<n;i+=2)
    {
        sum1+=c[i];
        sum2+=c[i+1];
    }
    ans = sum1+sum2;
    gettimeofday(&endtime_2,NULL);//end
    timeuse_2 = 1000000*(endtime_2.tv_sec-starttime_2.tv_sec)+(endtime_2.tv_usec-starttime_2.tv_usec);
    timeuse_2 /= 1000000;
    printf("timeuse_2 = %f\n",timeuse_2);
    cout<<ans<<endl;
    return 0;
}