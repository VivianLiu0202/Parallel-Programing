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

//递归函数
void recursion(int m)
{
    if(m==1) return ;
    else
    {
        for(int i=0;i<m/2;i++)
            c[i]+=c[m-i-1];
        m = m /2;
        recursion(m);
    }

}
int main()
{
    struct timeval starttime_3,endtime_3; 
    float timeuse_3;
    //----------------超标量优化-----------------计算n个数的和
    for(int i=0;i<n;i++)
        c[i] = i%20;
    int ans = 0;
   //递归
    /*
     *  1. 将给定元素两两相加，得到n/2个中间结果;
     *  2. 将上一步得到的中间结果两两相加，得到n/4个中间结果;
     *  3. 依此类推，log(n)个步骤后得到一个值即为最终结果。
     */
    //实现方式1：递归函数
    gettimeofday(&starttime_3,NULL);//begin
    recursion(n);
    gettimeofday(&endtime_3,NULL);//end
    timeuse_3 = 1000000*(endtime_3.tv_sec-starttime_3.tv_sec)+(endtime_3.tv_usec-starttime_3.tv_usec);
    timeuse_3 /= 1000000;
    printf("timeuse_3 = %f\n",timeuse_3);
    cout<<c[0]<<endl;


    //二重循环
    // gettimeofday(&starttime_3,NULL);//begin
    // for(int m = n;m>1;m/=2)
    //     for(int i = 0;i<m/2;i++)
    //         c[i] = c[i*2]+c[i*2+1];//相邻元素相加连续存储到数组最前面
    // gettimeofday(&endtime_3,NULL);//end
    // timeuse_3 = 1000000*(endtime_3.tv_sec-starttime_3.tv_sec)+(endtime_3.tv_usec-starttime_3.tv_usec);
    // timeuse_3 /= 1000000;
    // printf("timeuse_3 = %f\n",timeuse_3);
    // cout<<c[0]<<endl;
    return 0;
}