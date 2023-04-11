#include <iostream>
#include <windows.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <sys/time.h>
using namespace std;

float a[2000][2000], b[2000][2000], c[2000][2000], d[2000][2000],e[2000][2000];
void Initialize(int n)
{
    for (int i = 0; i < n; i++)
    {
        // 对角线元素初始化为1
        a[i][i] = 1.0;
        b[i][i] = 1.0;
        c[i][i] = 1.0;
        d[i][i] = 1.0;
        e[i][i] = 1.0;

        // 下三角元素初始化为0
        for (int j = 0; j < i; j++)
        {
            a[i][j] = 0;
            b[i][j] = 0;
            c[i][j] = 0;
            d[i][j] = 0;
            e[i][j] = 0;
        }

        // 上三角元素初始化为随机数
        for (int j = i + 1; j < n; j++)
        {
            a[i][j] = rand();
            b[i][j] = a[i][j];
            c[i][j] = a[i][j];
            d[i][j] = a[i][j];
            e[i][j] = a[i][j];
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
                e[i][j] += e[k][j];
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

void Gauss_Part1(int n) // 对第一个部分进行向量化的SSE并行算法,采用4路向量化
{
    int i, j, k;
    __m128 t1, t2, t3; // 定义4个向量寄存器
    for (k = 0; k < n; k++)
    {
        float temp[4] = {b[k][k], b[k][k], b[k][k], b[k][k]};
        t1 = _mm_loadu_ps(temp); // 加载到t1向量寄存器
        for (j = k + 1; j + 4 <= n; j += 4)
        {
            t2 = _mm_loadu_ps(b[k] + j); // 把内存中从b[k][j]开始的四个单精度浮点数加载到t2寄存器
            t3 = _mm_div_ps(t2, t1);     // 相除结果放到t3寄存器
            _mm_storeu_ps(b[k] + j, t3); // 把t3寄存器的值放回内存
        }
        for (j; j < n; j++)
            b[k][j] /= b[k][k];
        b[k][k] = 1.0;
        // complete the first part;

        for (i = k + 1; i < n; i++)
        {
            for (j = k + 1; j < n; j++)
                b[i][j] -= b[i][k] * b[k][j];
            b[i][k] = 0;
        }
    }
}

void Gauss_Part2(int n) // 对第二个部分进行SSE向量化并行算法
{
    int i, j, k;
    __m128 t1, t2, t3, t4;
    for (k = 0; k < n; k++)
    {
        for (j = k; j < n; j++)
            c[k][j] /= c[k][k];

        for (i = k + 1; i < n; i++)
        {
            float temp2[4] = {c[i][k], c[i][k], c[i][k], c[i][k]};
            t1 = _mm_loadu_ps(temp2);
            for (j = k + 1; j + 4 < n; j += 4)
            {
                t2 = _mm_loadu_ps(c[k] + j);
                t3 = _mm_loadu_ps(c[i] + j);
                t4 = _mm_mul_ps(t1, t2);
                t3 = _mm_sub_ps(t3, t4);
                _mm_storeu_ps(c[i] + j, t3);
            }
            for (j = j; j < n; j++)
                c[i][j] -= c[i][k] * c[k][j];
            c[i][k] = 0;
        }
    }
}

void Gauss_SSE_unaligned(int n)
{
    int i, j, k;
    __m128 t1, t2, t3, t4;
    for (k = 0; k < n; k++)
    {
        float temp[4] = {d[k][k], d[k][k], d[k][k], d[k][k]};
        t1 = _mm_loadu_ps(temp);
        for (j = k + 1; j + 4 <= n; j += 4)
        {
            t2 = _mm_loadu_ps(d[k] + j); // 把内存中从d[k][j]开始的四个单精度浮点数加载到t2寄存器
            t3 = _mm_div_ps(t2, t1);     // 相除结果放到t3寄存器
            _mm_storeu_ps(d[k] + j, t3); // 把t3寄存器的值放回内存
        }
        for (j; j < n; j++)
            d[k][j] /= d[k][k]; // 处理不能被4整除的
        d[k][k] = 1.0;

        for (i = k + 1; i < n; i++)
        {
            float temp2[4] = {d[i][k], d[i][k], d[i][k], d[i][k]};
            t1 = _mm_loadu_ps(temp2);
            for (j = k + 1; j + 4 < n; j += 4)
            {
                t2 = _mm_loadu_ps(d[k] + j);
                t3 = _mm_loadu_ps(d[i] + j);
                t4 = _mm_mul_ps(t1, t2);
                t3 = _mm_sub_ps(t3, t4);
                _mm_storeu_ps(d[i] + j, t3);
            }
            for (j = j; j < n; j++)
                d[i][j] -= d[i][k] * d[k][j];
            d[i][k] = 0;
        }
    }
}

void Gauss_SSE_aligned(int n)
{
    int i, j, k;
    __m128 t1, t2, t3, t4;
    for (k = 0; k < n; k++)
    {
        float temp[4] = {e[k][k], e[k][k], e[k][k], e[k][k]};
        t1 = _mm_load_ps(temp);
        j=k+1;
        while((k*n+j)%4!=0){
            e[k][j]=e[k][j]*1.0/e[k][k];
            j++;
        }
        for ( ; j + 4 <= n; j += 4)
        {
            t2 = _mm_load_ps(e[k] + j); // 把内存中从d[k][j]开始的四个单精度浮点数加载到t2寄存器
            t3 = _mm_div_ps(t2, t1);     // 相除结果放到t3寄存器
            _mm_store_ps(e[k] + j, t3); // 把t3寄存器的值放回内存
        }
        for (j; j < n; j++)
            e[k][j] /= e[k][k]; // 处理不能被4整除的
        e[k][k] = 1.0;

        for (i = k + 1; i < n; i++)
        {
            float temp2[4] = {e[i][k], e[i][k], e[i][k], e[i][k]};
            t1 = _mm_load_ps(temp2);
            j=k+1;
            while((k*n+j)%4!=0){//对齐操作
                e[i][j]=e[i][j]-e[k][j]*e[i][k];
                j++;
            }
            for (; j + 4 <= n; j += 4)
            {
                t2 = _mm_load_ps(e[k] + j);
                t3 = _mm_load_ps(e[i] + j);
                t4 = _mm_mul_ps(t1, t2);
                t3 = _mm_sub_ps(t3, t4);
                _mm_store_ps(e[i] + j, t3);
            }
            for ( ; j < n; j++) e[i][j] -= e[i][k] * e[k][j];
            e[i][k] = 0;
        }
    } 
}

void Print(int n, float m[][2000]) // 打印结果
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << m[i][j] << " ";
        }
        cout << endl;
    }
}

int main()
{
    int N = 1024;
    int count;
    int cycle;
    LARGE_INTEGER t1,t2,tc1,t3,t4,tc2;
    LARGE_INTEGER t5,t6,t7,t8,tc3,tc4,t9,t10,tc5;
    int ans = 0;
    for (int n = 2; n <= N; n *= 2)
    {
        ans++;
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


        cout<<"第"<<ans<<"次："<<endl;
        cout<<"常规算法："<<endl;
        QueryPerformanceFrequency(&tc1);
        QueryPerformanceCounter(&t1);
        while (count < cycle)
        {
            Gauss_Normal(n);
            count++;
        }
        QueryPerformanceCounter(&t2);
        cout<<n<<" "<<count<<" "<<((t2.QuadPart - t1.QuadPart)*1000.0 / tc1.QuadPart)<<"ms"<<endl;

        cout<<"优化第一个二重循环部分"<<endl;
        count=1;
        QueryPerformanceFrequency(&tc3);
        QueryPerformanceCounter(&t5);
        while (count < cycle)
        {
            Gauss_Part1(n);
            count++;
        }
        QueryPerformanceCounter(&t6);
        cout<<n<<" "<<count<<" "<<((t6.QuadPart - t5.QuadPart)*1000.0 / tc3.QuadPart)<<"ms"<<endl;


        cout<<"优化第二个三重循环部分"<<endl;
        count=1;
        QueryPerformanceFrequency(&tc4);
        QueryPerformanceCounter(&t7);
        while (count < cycle)
        {
            Gauss_Part2(n);
            count++;
        }
        QueryPerformanceCounter(&t8);
        cout<<n<<" "<<count<<" "<<((t8.QuadPart - t7.QuadPart)*1000.0 / tc4.QuadPart)<<"ms"<<endl;

        cout<<"SSE_unaligned"<<endl;
        count = 1;
        QueryPerformanceFrequency(&tc2);
        QueryPerformanceCounter(&t3);
        while (count < cycle)
        {
            Gauss_SSE_unaligned(n);
            count++;
        }
        QueryPerformanceCounter(&t4);
        cout<<n<<" "<<count<<" "<<((t4.QuadPart - t3.QuadPart)*1000.0 / tc2.QuadPart)<<"ms"<<endl;

        cout<<"SSE_aligned"<<endl;
        count = 1;
        QueryPerformanceFrequency(&tc5);
        QueryPerformanceCounter(&t9);
        while (count < cycle)
        {
            Gauss_SSE_aligned(n);
            count++;
        }
        QueryPerformanceCounter(&t10);
        cout<<n<<" "<<count<<" "<<((t10.QuadPart - t9.QuadPart)*1000.0 / tc5.QuadPart)<<"ms"<<endl;

        cout<<endl<<endl;
    }

    return 0;
}