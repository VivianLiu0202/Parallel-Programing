#include <iostream>
#include<windows.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <sys/time.h>
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
            for (int j = k + 1; j < n; j++)
                a[i][j] -= a[i][k] * a[k][j]; // 可以进行向量化
            a[i][k] = 0;
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
    LARGE_INTEGER t5,t6,t7,t8,tc3,tc4;
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
        count = 1;
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
    }
    cout<<"finish"<<endl;
    return 0;
}