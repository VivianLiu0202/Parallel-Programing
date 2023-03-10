#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace std;
#define n 10000
int sum[n];
int a[n];//1*n matrix
int b[n][n];//n*n matrix
// void print1()
// {
//     for(int i=0;i<n;i++)
//         cout<<sum[i]<<' ';
// }

int main() {
    //std::srand((unsigned)time(NULL));
    //-----------------矩阵内积------------------
    //建立100*100随机数矩阵
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
        {
            b[i][j] = i+j;
        }
    for(int i=0;i<n;i++)
        a[i] = rand()%10; 
    struct timeval starttime_1,endtime_1;
    struct timeval starttime_2,endtime_2;
    
    float timeuse_1,timeuse_2;


    //-----------------平凡算法------------------
    memset(sum,0,sizeof(sum));
    gettimeofday(&starttime_1,NULL);//begin
    for(int i=0;i<n;i++)
    {
        sum[i] = 0.0;
        for(int j=0;j<n;j++)
            sum[i]+=b[j][i]*a[j];
    }
    gettimeofday(&endtime_1,NULL);//end
    timeuse_1 = 1000000*(endtime_1.tv_sec-starttime_1.tv_sec)+(endtime_1.tv_usec-starttime_1.tv_usec);
    timeuse_1 /= 1000000;
    printf("timeuse = %f\n",timeuse_1);

    //-----------------优化算法------------------
    //访存模式与行主存储匹配，具有很好的空间局部性
    memset(sum,0,sizeof(sum));
    gettimeofday(&starttime_2,NULL);//begin
    for(int i=0;i<n;i++)
        sum[i] = 0.0;
    for(int j=0;j<n;j++)
        for(int i=0;i<n;i++)
            sum[i]+=b[j][i]*a[j];
    gettimeofday(&endtime_2,NULL);//end
    timeuse_2 = 1000000*(endtime_2.tv_sec-starttime_2.tv_sec)+(endtime_2.tv_usec-starttime_2.tv_usec);
    timeuse_2 /= 1000000;
    printf("timeuse = %f\n",timeuse_2);        
    return 0;
}