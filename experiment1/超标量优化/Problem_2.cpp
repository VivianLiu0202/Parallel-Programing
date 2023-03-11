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
unsigned long long int c[n];//n个数
unsigned long long int ans = 0;
int LOOP = 10;
void init()
{
    for(unsigned i =0;i<n;i++)
        c[i] = i;
}

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

    init();
    //-----------------平凡算法-----------------
    struct timeval starttime_1,endtime_1;
    gettimeofday(&starttime_1,NULL);//begin
    for(int k=0;k<=LOOP;k++)
    {
        unsigned sum = 0;
        for (int i = 0; i < n - 1; i+=2)
            sum += c[i], sum += c[i+1]; 
    }
    gettimeofday(&endtime_1,NULL);//end
    cout<<"common used time:"<<((endtime_1.tv_sec-starttime_1.tv_sec)*1000000+(endtime_1.tv_usec-starttime_1.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
   
    //-----------------优化算法------------------
    //多链路式
    struct timeval starttime_2,endtime_2;
    unsigned long long int sum1 = 0;
    unsigned long long int sum2 = 0;
    gettimeofday(&starttime_2,NULL);//begin
    for(int k = 1;k<=LOOP;k++)
    {
        for(int i=0;i<n;i+=2)
        {
            sum1+=c[i];
            sum2+=c[i+1];
        }
    }
    
    ans = sum1+sum2;
    gettimeofday(&endtime_2,NULL);//end
    cout<<"optimize used time:"<<((endtime_2.tv_sec-starttime_2.tv_sec)*1000000+(endtime_2.tv_usec-starttime_2.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
    
   //递归
    /*
     *  1. 将给定元素两两相加，得到n/2个中间结果;
     *  2. 将上一步得到的中间结果两两相加，得到n/4个中间结果;
     *  3. 依此类推，log(n)个步骤后得到一个值即为最终结果。
     */
    //实现方式1：递归函数
    struct timeval starttime_3,endtime_3; 
    gettimeofday(&starttime_3,NULL);//begin
    for(int k=1;k<=LOOP;k++)
    {
        recursion(n);
    }
    gettimeofday(&endtime_3,NULL);//end
    cout<<"recursion used time:"<<((endtime_3.tv_sec-starttime_3.tv_sec)*1000000+(endtime_3.tv_usec-starttime_3.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
    return 0;




}