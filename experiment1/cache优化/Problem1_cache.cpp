#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace std;
const int n = 1000;
unsigned long long int sum[n];
unsigned long long int a[n];//1*n matrix
unsigned long long int b[n][n];//n*n matrix
int LOOP = 1;
void init()
{
        for(int i=0;i<n;i++)
    {
        a[i] = i;
        for(int j=0;j<n;j++)
        {
            b[i][j] = i+j;
        }
    }
}
int main() {
    init();
    struct timeval starttime_2,endtime_2;
    //-----------------优化算法------------------
    //访存模式与行主存储匹配，具有很好的空间局部性
    memset(sum,0,sizeof(sum));
    gettimeofday(&starttime_2,NULL);//begin
    for(int k = 1;k<=LOOP;k++)
    {
        for(int i=0;i<n;i++)
            sum[i] = 0.0;
        for(int j=0;j<n;j++)
            for(int i=0;i<n;i++)
                sum[i]+=b[j][i]*a[j];
    }
    gettimeofday(&endtime_2,NULL);//end      
    cout<<"cache used time:"<<((endtime_2.tv_sec-starttime_2.tv_sec)*1000000+(endtime_2.tv_usec-starttime_2.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
    return 0;
}