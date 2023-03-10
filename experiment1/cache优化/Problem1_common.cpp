#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace std;
const int n = 10;
unsigned long long int sum[n];
unsigned long long int a[n];//1*n matrix
unsigned long long int b[n][n];//n*n matrix

int LOOP = 10000;
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
    //-----------------矩阵内积------------------
    //建立100*100随机数矩阵
    init();
    struct timeval starttime_1,endtime_1;
    //-----------------平凡算法------------------
    memset(sum,0,sizeof(sum));
    gettimeofday(&starttime_1,NULL);//begin
    for(int k = 1;k<=LOOP;k++)
    {
        for(int i=0;i<n;i++)
        {
            sum[i] = 0.0;
            for(int j=0;j<n;j++)
                sum[i]+=b[j][i]*a[j];
        }
    }
    gettimeofday(&endtime_1,NULL);//end
    cout<<"common used time:"<<((endtime_1.tv_sec-starttime_1.tv_sec)*1000000+(endtime_1.tv_usec-starttime_1.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
}

