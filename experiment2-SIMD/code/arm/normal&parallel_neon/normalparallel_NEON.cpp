#include<iostream>
#include<sys/time.h>
#include<arm_neon.h>

using namespace std;
float a[2000][2000], b[2000][2000], c[2000][2000], d[2000][2000];
void Initialize(int n)
{
    for (int i = 0; i < n; i++)
    {
        // 对角线元素初始化为1
        a[i][i] = 1.0;
        b[i][i] = 1.0;
        c[i][i] = 1.0;
        d[i][i] = 1.0;

        // 下三角元素初始化为0
        for (int j = 0; j < i; j++)
        {
            a[i][j] = 0;
            b[i][j] = 0;
            c[i][j] = 0;
            d[i][j] = 0;
        }

        // 上三角元素初始化为随机数
        for (int j = i + 1; j < n; j++)
        {
            a[i][j] = rand();
            b[i][j] = a[i][j];
            c[i][j] = a[i][j];
            d[i][j] = a[i][j];
        }
    }
    for (int k = 0; k < n; k++)
    {
        for (int i = k + 1; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                // 最终每一行的值是上一行的值与这一行的值之和
                a[i][j] += a[k][j];
                b[i][j] += b[k][j];
                c[i][j] += c[k][j];
                d[i][j] += d[k][j];
            }
        }
    }
}

// 串行算法
void Gauss_Normal(int n)
{
    for (int k = 0; k < n; k++)
    {
        float temp = a[k][k];
        for (int j = k; j < n; j++)
            a[k][j] /= temp; // 可以进行向量化，用SIMD并行优化
        for (int i = k + 1; i < n; i++)
        {
            // float temp2 = a[i][k];
            for (int j = k + 1; j < n; j++)
                a[i][j] -= a[i][k] * a[k][j]; // 可以进行向量化
            a[i][k] = 0;
        }
    }
}

void Gauss_para1(int n){ //对第一个部分进行向量化的SSE并行算法
    int i,j,k;
    float32x4_t t1,t2,t3,t4; //定义4个向量寄存器
    for(k=0;k<n;k++)
    {
        float32_t tmp[4]={b[k][k],b[k][k],b[k][k],b[k][k]};
        t1=vld1q_f32(tmp); //加载到t1向量寄存器
        for(j=k+1;j+4<=n;j+=4)
        {
            t2=vld1q_f32(b[k]+j); //把内存中从B[k][j]开始的四个单精度浮点数加载到t2寄存器
            t3=vdivq_f32(t2,t1); //相除结果放到t3寄存器
            vst1q_f32(b[k]+j,t3); //把t3寄存器的值放回内存
        }
        for(j;j<n;j++) //处理剩下的不能被4整除的
            b[k][j]/=b[k][k];
        b[k][k]=1.0;
        //以上完成了对第一个部分的向量化

        for(i=k+1;i<n;i++)
        {
			float temp2=b[i][k];
			for(j=k+1;j<n;j++)
				b[i][j] -= temp2*b[k][j];//可以进行向量化，用SIMD 扩展指令进行并行优化
			b[i][k]=0;
        }
    }
}
void Gauss_Part2(int n){ //对第二个部分进行向量化的SSE并行算法
    int i,j,k;
    float32x4_t t1,t2,t3,t4; //定义4个向量寄存器
    for(k=0;k<n;k++)
    {
        float tmp=c[k][k];
		for(j=k;j<n;j++)
			c[k][j]/=tmp;//可以进行向量化，用SIMD 扩展指令进行并行优化

        for(i=k+1;i<n;i++)
        {
            float32_t tmp2[4]={c[i][k],c[i][k],c[i][k],c[i][k]};
            t1=vld1q_f32(tmp2);
            for(j=k+1;j+4<=n;j+=4)
            {
                t2=vld1q_f32(c[k]+j);
                t3=vld1q_f32(c[i]+j);
                t4=vmulq_f32(t1,t2);
                t3=vsubq_f32(t3,t4);
                vst1q_f32(c[i]+j,t3);
            }
            for(j=j;j<n;j++)
                c[i][j]-=c[i][k]*c[k][j];
            c[i][k]=0;
        }
    }
}
void Gauss_NEON_unaligned(int n)
{
    int i,j,k;
    float32x4_t t1,t2,t3,t4; //定义4个向量寄存器
    for(k=0;k<n;k++)
    {
        float32_t temp[4]={d[k][k],d[k][k],d[k][k],d[k][k]};
        t1=vld1q_f32(temp); //加载到t1向量寄存器
        for(j=k+1;j+4<=n;j+=4)
        {
            t2=vld1q_f32(d[k]+j); //把内存中从B[k][j]开始的四个单精度浮点数加载到t2寄存器
            t3=vdivq_f32(t2,t1); //相除结果放到t3寄存器
            vst1q_f32(d[k]+j,t3); //把t3寄存器的值放回内存
        }
        for(j;j<n;j++) //处理剩下的不能被4整除的
            d[k][j]/=d[k][k];
        d[k][k]=1.0;
        //以上完成了对第一个部分的向量化

        for(i=k+1;i<n;i++)
        {
            float32_t tmp2[4]={d[i][k],d[i][k],d[i][k],d[i][k]};
            t1=vld1q_f32(tmp2);
            for(j=k+1;j+4<=n;j+=4)
            {
                t2=vld1q_f32(d[k]+j);
                t3=vld1q_f32(d[i]+j);
                t4=vmulq_f32(t1,t2);
                t3=vsubq_f32(t3,t4);
                vst1q_f32(d[i]+j,t3);
            }
            for(j=j;j<n;j++)
                d[i][j]-=d[i][k]*d[k][j];
            d[i][k]=0;
        }
    }
}

void Gauss_NEON_aligned(int n)\
{
    int i,j,k;
    float32x4_t t1,t2,t3,t4; //定义4个向量寄存器
    for(k=0;k<n;k++)
    {
        float32x4_t t1=vmovq_n_f32(d[k][k]);
        j = k+1; 

        while((k*n+j)%4!=0){//对齐操作
            d[k][j]=d[k][j]*1.0/d[k][k];
            j++;
        }
        for(;j+4<=n;j+=4)
        {
            t2=vld1q_f32(d[k]+j); //把内存中从B[k][j]开始的四个单精度浮点数加载到t2寄存器
            t3=vdivq_f32(t2,t1); //相除结果放到t3寄存器
            vst1q_f32(d[k]+j,t3); //把t3寄存器的值放回内存
        }
        for(j;j<n;j++) //处理剩下的不能被4整除的
            d[k][j]/=d[k][k];
        d[k][k]=1.0;
        //以上完成了对第一个部分的向量化

        for(i=k+1;i<n;i++)
        {
            float32x4_t t1 =  vmovq_n_f32(d[i][k]);
            j=k+1;
            while((k*n+j)%4!=0){//对齐操作
                d[i][j]=d[i][j]-d[k][j]*d[i][k];
                j++;
            }
            for(;j+4<=n;j+=4)
            {
                t2=vld1q_f32(d[k]+j);
                t3=vld1q_f32(d[i]+j);
                t4=vmulq_f32(t1,t2);
                t3=vsubq_f32(t3,t4);
                vst1q_f32(d[i]+j,t3);
            }
            for(j=j;j<n;j++) d[i][j] -= d[i][k]*d[k][j];
            d[i][k]=0.0;
        }
    }
}

void Print(int n,float m[][2000]){//打印结果
	int i,j;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++)
			cout<<m[i][j]<<" ";
		cout<<endl;
	}
}

int main()
{
    int N = 1024;
    int count,cycle;
    struct timeval head1, head2, head3, head4;
    struct timeval tail1, tail2, tail3, tail4;
    for (int n = 2; n <= N; n *= 2)
    {
        Initialize(n);
        count = 1;
        if (n <= 30)
            cycle = 1000;
        else if (n <= 70)
            cycle = 100;
        else if (n <= 300)
            cycle = 50;
        else
            cycle = 10;

        gettimeofday(&head1, NULL);
        while (count < cycle)
        {
            Gauss_NEON_aligned(n);
            count++;
        }

        gettimeofday(&tail1, NULL);
        cout << n << " " << count << " " << (tail1.tv_sec - head1.tv_sec) * 1000.0 + (tail1.tv_usec - head1.tv_usec) / 1000.0 << "ms" << endl;
        count = 1;
        gettimeofday(&head2, NULL);

        while (count < cycle)
        {
            Gauss_NEON_unaligned(n);
            count++;
        }
        gettimeofday(&tail2, NULL);
        cout << n << " " << count << " " << (tail2.tv_sec - head2.tv_sec) * 1000.0 + (tail2.tv_usec - head2.tv_usec) / 1000.0 << "ms" << endl;
    }
    return 0;
}