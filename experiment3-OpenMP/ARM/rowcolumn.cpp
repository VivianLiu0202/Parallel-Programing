#include <iostream>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include <arm_neon.h>
#include <pthread.h> // pthread
#include <semaphore.h>
#include </opt/homebrew/Cellar/libomp/16.0.2/include/omp.h>
using namespace std;

int NUM_THREADS = 8;
// ============================================== pthread 线程控制变量 ==============================================
typedef struct
{
    int t_id;
} threadParam_t;

sem_t sem_Division;
pthread_barrier_t barrier;
// ============================================== 运算变量 ==============================================
int N;
const int L = 100;
const int LOOP = 50;
float **Data;
float **matrix;

ofstream res_stream;

void init_data()
{
    Data = new float *[N], matrix = new float *[N];
    for (int i = 0; i < N; i++)
        Data[i] = new float[N], matrix[i] = new float[N];
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++)
            Data[i][j] = rand() * 1.0 / RAND_MAX * L;
    for (int i = 0; i < N - 1; i++)
        for (int j = i + 1; j < N; j++)
            for (int k = 0; k < N; k++)
                Data[j][k] += Data[i][k];
}

// 用data初始化matrix，保证每次进行计算的数据是一致的
void init_matrix()
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = Data[i][j];
}
void calculate_openmp_row()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}
// 按列划分
void calculate_openmp_column()
{
    int i, j, k;
#pragma omp parallel num_threads(NUM_THREADS), default(none), private(i, j, k), shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp for schedule(simd \
                         : guided)
        for (j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
            for (i = k + 1; i < N; i++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
        }
#pragma omp single
        {
            matrix[k][k] = 1;
            for (i = k + 1; i < N; i++)
            {
                matrix[i][k] = 0;
            }
        }
    }
}
// 打印矩阵
void print_matrix()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}
void test(int n)
{
    N = n;
    cout << "=================================== " << N << " ===================================" << endl;
    res_stream << N;
    struct timeval start;
    struct timeval end;
    float time = 0;
    init_data();
    // ====================================== openmp_row ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_row();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_row:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== openmp_column ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_column();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_column:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    res_stream << endl;
    //    print_matrix();
}

int main()
{
    res_stream.open("result.csv", ios::out);
    for (int i = 100; i <= 1000; i += 100)
        test(i);
    for (int i = 1000; i <= 3000; i += 500)
        test(i);
    res_stream.close();
    return 0;
}
