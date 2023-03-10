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
    struct timeval starttime_3,endtime_3;
    //-----------------优化算法------------------
    //访存模式与行主存储匹配，具有很好的空间局部性
    memset(sum,0,sizeof(sum));
    gettimeofday(&starttime_3,NULL);//begin
    for(int k= 1;k<=LOOP;k++)
    {
        for(int i=0;i<n;i+=20)
        {
            int tmp0=0,tmp1=0,tmp2=0,tmp3=0,tmp4=0,tmp5=0,tmp6=0,tmp7=0,tmp8=0,tmp9=0,tmp10=0,tmp11=0,tmp12=0,tmp13=0,tmp14=0,tmp15=0,tmp16=0,tmp17=0,tmp18=0,tmp19=0; 
            for(int j=0;j<n;j++)
            {
                    tmp0+=a[i+0]*b[i+0][j];
                    tmp1+=a[i+1]*b[i+1][j];
                    tmp2+=a[i+2]*b[i+2][j];
                    tmp3+=a[i+3]*b[i+3][j];
                    tmp4+=a[i+4]*b[i+4][j];
                    tmp5+=a[i+5]*b[i+5][j];
                    tmp6+=a[i+6]*b[i+6][j];
                    tmp6+=a[i+6]*b[i+6][j];
                    tmp7+=a[i+7]*b[i+7][j];
                    tmp8+=a[i+8]*b[i+8][j];
                    tmp9+=a[i+9]*b[i+9][j];
                    tmp10+=a[i+10]*b[i+10][j];
                    tmp11+=a[i+11]*b[i+11][j];
                    tmp12+=a[i+12]*b[i+12][j];
                    tmp13+=a[i+13]*b[i+13][j];
                    tmp14+=a[i+14]*b[i+14][j];
                    tmp15+=a[i+15]*b[i+15][j];
                    tmp16+=a[i+16]*b[i+16][j];
                    tmp17+=a[i+17]*b[i+17][j];
                    tmp18+=a[i+18]*b[i+18][j];
                    tmp19+=a[i+19]*b[i+19][j];
            }
                sum[i+0]=tmp0;
                sum[i+1]=tmp1;
                sum[i+2]=tmp2;
                sum[i+3]=tmp3;
                sum[i+4]=tmp4;
                sum[i+5]=tmp5;
                sum[i+6]=tmp6;
                sum[i+7]=tmp7;
                sum[i+8]=tmp8;
                sum[i+9]=tmp9;
                sum[i+10]=tmp10;
                sum[i+11]=tmp11;
                sum[i+12]=tmp12;
                sum[i+13]=tmp13;
                sum[i+14]=tmp14;
                sum[i+15]=tmp15;
                sum[i+16]=tmp16;
                sum[i+17]=tmp17;
                sum[i+18]=tmp18;
                sum[i+19]=tmp19;
        }
    }
    gettimeofday(&endtime_3,NULL);//end      
    cout<<"unroll used time:"<<((endtime_3.tv_sec-starttime_3.tv_sec)*1000000+(endtime_3.tv_usec-starttime_3.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
    return 0;
}