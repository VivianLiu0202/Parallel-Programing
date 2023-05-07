#include <iostream>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include <arm_neon.h>
#include <pthread.h> // pthread
#include <semaphore.h>
#include </opt/homebrew/Cellar/libomp/16.0.2/include/omp.h>
using namespace std;
ofstream res_stream;

int NUM_THREADS = 8;
const int num_threads=7;
// ============================================== pthread 线程控制变量 ==============================================
typedef struct
{
    int t_id;
} threadParam_t;

// ============================================== 运算变量 ==============================================
int N;
const int L = 100;
const int LOOP = 50;
float **Data;
float **matrix;

//信号量定义
sem_t sem_leader;
sem_t sem_Division[num_threads-1];
sem_t sem_Elimination[num_threads-1];

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

// 串行算法
void calculate_serial()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// SIMD并行算法
void calculate_SIMD()
{
    for (int k = 0; k < N; k++)
    {
        float32x4_t Akk = vmovq_n_f32(matrix[k][k]);
        int j;
        for (j = k + 1; j + 3 < N; j += 4)
        {
            float32x4_t Akj = vld1q_f32(matrix[k] + j);
            Akj = vdivq_f32(Akj, Akk);
            vst1q_f32(matrix[k] + j, Akj);
        }
        for (; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            float32x4_t Aik = vmovq_n_f32(matrix[i][k]);
            for (j = k + 1; j + 3 < N; j += 4)
            {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                float32x4_t Aij = vld1q_f32(matrix[i] + j);
                float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
                Aij = vsubq_f32(Aij, AikMulAkj);
                vst1q_f32(matrix[i] + j, Aij);
            }
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 单独使用openmp进行simd优化
void calculate_openmp_single_simd()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(1) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
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
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

void *threadFunc2(void* param){//循环划分
    threadParam_t *p=(threadParam_t*)param;
    int t_id=p->t_id; //线程编号
    for(int k=0;k<N;k++)
    {
        if(t_id==0)
        {
            float32x4_t Akk = vmovq_n_f32(matrix[k][k]);
            int j;
            for (j = k + 1; j + 3 < N; j += 4)
            {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                Akj = vdivq_f32(Akj, Akk);
                vst1q_f32(matrix[k] + j, Akj);
            }
            for (; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        else
        {
            sem_wait(&sem_Division[t_id-1]);
        }

        // t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if(t_id==0)
        {
            for(int i=0;i<num_threads-1;i++)
                sem_post(&sem_Division[i]);
		}

        //循环划分任务
        float32x4_t t1,t2,t3,t4; //定义4个向量寄存器
        int j;
		for(int i=k+1+t_id;i<N;i+=num_threads)
		{
			float32x4_t t1 = vmovq_n_f32(matrix[i][k]);
            for(j=k+1;j+4<=N;j+=4)
            {
                t2=vld1q_f32(matrix[k]+j);
                t3=vld1q_f32(matrix[i]+j);
                t4=vmulq_f32(t1,t2);
                t3=vsubq_f32(t3,t4);
                vst1q_f32(matrix[i]+j,t3);
            }
            for(j=j;j<N;j++)
                matrix[i][j]-=matrix[i][k]*matrix[k][j];
            matrix[i][k]=0.0;
		}

		if(t_id==0)
        {
            for(int i=0;i<num_threads-1;i++)
                sem_wait(&sem_leader);// 等待其它 worker 完成消去
            for(int i=0;i<num_threads-1;i++)
                sem_post(&sem_Elimination[i]);// 通知其它 worker 进入下一轮
        }
        else
        {
            sem_post(&sem_leader);// 通知 leader, 已完成消去任务
            sem_wait(&sem_Elimination[t_id-1]);// 等待通知，进入下一轮
        }
	}
	pthread_exit(NULL);

}

void calculate_pthread(){
	int i;
    //初始化信号量
	sem_init(&sem_leader,0,0);
	for(i=0;i<num_threads-1;i++)
    {
        sem_init(&sem_Division[i],0,0);
        sem_init(&sem_Elimination[i],0,0);
	}
    //创建线程
	pthread_t *handles=new pthread_t[num_threads];//创建对应的handle
    threadParam_t *param=new threadParam_t[num_threads];//创建对应的线程数据结构
    for(int t_id=0;t_id<num_threads;t_id++)
    {
        param[t_id].t_id=t_id;
    // }
    // for(int t_id=0;t_id<num_threads;t_id++)
    // {
        pthread_create(&handles[t_id],NULL,threadFunc2,&param[t_id]);
    }
    for(int t_id=0;t_id<num_threads;t_id++)
    {
        pthread_join(handles[t_id],NULL);
    }
	sem_destroy(sem_Division);
	sem_destroy(sem_Elimination);
	sem_destroy(&sem_leader);
}
// 静态数据划分
void calculate_openmp_schedule_static()
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
                         : static)
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
    // ====================================== serial ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_serial();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "serial:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== SIMD ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_SIMD();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "SIMD:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== openmp_single_SIMD ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_single_simd();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_single_SIMD:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== pthread ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_pthread();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "pthread:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== openmp_schedule_static ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_schedule_static();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_schedule_static:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();

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