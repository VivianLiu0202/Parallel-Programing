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
    struct timeval starttime_2,endtime_2;
    //-----------------优化算法------------------
    //多链路式
    unsigned long long int sum1 = 0;
    unsigned long long int sum2 = 0;
    gettimeofday(&starttime_2,NULL);//begin
    for(int k = 1;k<=LOOP;k++)
    {
	unsigned long long int sum1 = 0;
   	unsigned long long int sum2 = 0;
        for(int i=0;i<n;i+=2)
        {
            sum1+=c[i];
            sum2+=c[i+1];
        }
	unsigned long long int and = sum1+sum2;
    }

    gettimeofday(&endtime_2,NULL);//end
    cout<<"optimize used time:"<<((endtime_2.tv_sec-starttime_2.tv_sec)*1000000+(endtime_2.tv_usec-starttime_2.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
    return 0;
}