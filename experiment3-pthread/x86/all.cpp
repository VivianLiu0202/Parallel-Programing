#include<iostream>
#include<sys/time.h>
#include<pthread.h>
#include<stdlib.h>
#include<semaphore.h>
#include<semaphore.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<immintrin.h>
#include<ammintrin.h>
#include<cmath>
using namespace std;
float Data[3000][3000],matrix[3000][3000];
int n;
//线程数据结构定义
typedef struct
{
    int t_id;
} threadParam_t;

sem_t sem_Division;
pthread_barrier_t barrier;
pthread_mutex_t task;
int index = 0;

const int THREAD_NUM = 8;

// 初始化data，保证每次数据都是一致的
void init_data()
{
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
            Data[i][j] = rand() * 1.0;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            for (int k = 0; k < n; k++)
                Data[j][k] += Data[i][k];
}

// 用data初始化matrix，保证每次进行计算的数据是一致的
void init_matrix()
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = Data[i][j];
}

// 串行算法
void Gauss_Normal(int n)
{
    for (int k = 0; k < n; k++)
    {
        float temp = matrix[k][k];
        for (int j = k; j < n; j++)
            matrix[k][j] /= temp; // 可以进行向量化，用SIMD并行优化
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
                matrix[i][j] -= matrix[i][k] * matrix[k][j]; // 可以进行向量化
            matrix[i][k] = 0;
        }
    }
}

void Gauss_SSE()
{
    int i, j, k;
    __m128 t1, t2, t3, t4;
    for (k = 0; k < n; k++)
    {
        float temp[4] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm_loadu_ps(temp);
        for (j = k + 1; j + 4 <= n; j += 4)
        {
            t2 = _mm_loadu_ps(matrix[k] + j); // 把内存中从d[k][j]开始的四个单精度浮点数加载到t2寄存器
            t3 = _mm_div_ps(t2, t1);     // 相除结果放到t3寄存器
            _mm_storeu_ps(matrix[k] + j, t3); // 把t3寄存器的值放回内存
        }
        for (j; j < n; j++)
            matrix[k][j] /= matrix[k][k]; // 处理不能被4整除的
        matrix[k][k] = 1.0;
        for (i = k + 1; i < n; i++)
        {
            float temp2[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
            t1 = _mm_loadu_ps(temp2);
            for (j = k + 1; j + 4 < n; j += 4)
            {
                t2 = _mm_loadu_ps(matrix[k] + j);
                t3 = _mm_loadu_ps(matrix[i] + j);
                t4 = _mm_mul_ps(t1, t2);
                t3 = _mm_sub_ps(t3, t4);
                _mm_storeu_ps(matrix[i] + j, t3);
            }
            for (j = j; j < n; j++)
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}

// pthread_discrete 线程函数
void *threadFunc_discrete(void *param)
{
    threadParam_t *thread_param_t = (threadParam_t *)param;
    int t_id = thread_param_t->t_id;
    int i,j,k;
    __m128_u t1,t2,t3,t4;
    for (int k = 0; k < n; k++)
    {
        // 如果当前是0号线程，则进行除法操作，其余线程处于等待状态
        if (t_id == 0)
        {
            float temp[4] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
            t1 = _mm_loadu_ps(temp);
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                t2 = _mm_loadu_ps(matrix[k] + j); // 把内存中从d[k][j]开始的四个单精度浮点数加载到t2寄存器
                t3 = _mm_div_ps(t2, t1);     // 相除结果放到t3寄存器
                _mm_storeu_ps(matrix[k] + j, t3); // 把t3寄存器的值放回内存
            }
            for (j; j < n; j++)
                matrix[k][j] /= matrix[k][k]; // 处理不能被4整除的
            matrix[k][k] = 1.0;
        }
        else
        {
            sem_wait(&sem_Division);
        }

        // 除法操作完成后，如果是0号线程，则需要唤醒其他线程
        if (t_id == 0)
        {
            for (int i = 1; i < THREAD_NUM; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            // 循环划分任务
            for (int i = k + t_id; i < n; i += (THREAD_NUM - 1))
            {
                float temp2[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
                t1 = _mm_loadu_ps(temp2);
                for (j = k + 1; j + 4 < n; j += 4)
                {
                    t2 = _mm_loadu_ps(matrix[k] + j);
                    t3 = _mm_loadu_ps(matrix[i] + j);
                    t4 = _mm_mul_ps(t1, t2);
                    t3 = _mm_sub_ps(t3, t4);
                    _mm_storeu_ps(matrix[i] + j, t3);
                }
                for (j = j; j < n; j++)
                    matrix[i][j] -= matrix[i][k] * matrix[k][j];
                matrix[i][k] = 0;
            }
        }
        // 所有线程准备进入下一轮
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread_discrete 并行算法
void calculate_pthread_discrete()
{
    // 信号量初始化
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, THREAD_NUM);

    // 创建线程
    pthread_t threads[THREAD_NUM];
    threadParam_t thread_param_t[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc_discrete, (void *)(&thread_param_t[i]));
    }

    // 加入执行线程
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // 销毁信号量
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
}


// pthread_discrete 线程函数
void *threadFunc_continuous(void *param)
{
    threadParam_t *thread_param_t = (threadParam_t *)param;
    int t_id = thread_param_t->t_id;
    int i,j,k;
    __m128_u t1,t2,t3,t4;
    for (int k = 0; k < n; k++)
    {
        // 如果当前是0号线程，则进行除法操作，其余线程处于等待状态
        if (t_id == 0)
        {
            float temp[4] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
            t1 = _mm_loadu_ps(temp);
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                t2 = _mm_loadu_ps(matrix[k] + j); // 把内存中从d[k][j]开始的四个单精度浮点数加载到t2寄存器
                t3 = _mm_div_ps(t2, t1);     // 相除结果放到t3寄存器
                _mm_storeu_ps(matrix[k] + j, t3); // 把t3寄存器的值放回内存
            }
            for (j; j < n; j++)
                matrix[k][j] /= matrix[k][k]; // 处理不能被4整除的
            matrix[k][k] = 1.0;
        }
        else
        {
            sem_wait(&sem_Division);
        }

        // 除法操作完成后，如果是0号线程，则需要唤醒其他线程
        if (t_id == 0)
        {
            for (int i = 1; i < THREAD_NUM; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            int L =  ceil((n-k)*1.0 / (THREAD_NUM - 1));
            // 循环划分任务
            for (int i = k + (t_id - 1) * L + 1; i < n && i < k + t_id * L + 1 ; i++)
            {
                float temp2[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
                t1 = _mm_loadu_ps(temp2);
                for (j = k + 1; j + 4 < n; j += 4)
                {
                    t2 = _mm_loadu_ps(matrix[k] + j);
                    t3 = _mm_loadu_ps(matrix[i] + j);
                    t4 = _mm_mul_ps(t1, t2);
                    t3 = _mm_sub_ps(t3, t4);
                    _mm_storeu_ps(matrix[i] + j, t3);
                }
                for (j = j; j < n; j++)
                    matrix[i][j] -= matrix[i][k] * matrix[k][j];
                matrix[i][k] = 0;
            }
        }
        // 所有线程准备进入下一轮
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread_continuous 并行算法
void calculate_pthread_continuous()
{
    // 信号量初始化
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, THREAD_NUM);
    // 创建线程
    pthread_t threads[THREAD_NUM];
    threadParam_t thread_param_t[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc_continuous, (void *)(&thread_param_t[i]));
    }
    // 加入执行线程
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_join(threads[i], NULL);
    }
    // 销毁信号量
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
}

// pthread_dynamic 线程函数
void * threadFunc_dynamic(void *param)
{
    threadParam_t *thread_param_t = (threadParam_t *)param;
    int t_id = thread_param_t->t_id;
    int i,j,k;
    __m128_u t1,t2,t3,t4;
    for (int k = 0; k < n; k++)
    {
        // 如果当前是0号线程，则进行除法操作，其余线程处于等待状态
        if (t_id == 0)
        {
            float temp[4] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
            t1 = _mm_loadu_ps(temp);
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                t2 = _mm_loadu_ps(matrix[k] + j); // 把内存中从d[k][j]开始的四个单精度浮点数加载到t2寄存器
                t3 = _mm_div_ps(t2, t1);     // 相除结果放到t3寄存器
                _mm_storeu_ps(matrix[k] + j, t3); // 把t3寄存器的值放回内存
            }
            for (j; j < n; j++)
                matrix[k][j] /= matrix[k][k]; // 处理不能被4整除的
            matrix[k][k] = 1.0;
            index = k + 1;
        }
        else
        {
            sem_wait(&sem_Division);
        }

        // 除法操作完成后，如果是0号线程，则需要唤醒其他线程
        if (t_id == 0)
        {
            for (int i = 1; i < THREAD_NUM; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            while(index < n)
            {
                pthread_mutex_lock(&task);
                int i = index;
                if( i < n )
                {
                    index++;
                    pthread_mutex_unlock(&task);
                }
                else
                {
                    pthread_mutex_unlock(&task);
                    break;
                }
                float temp2[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
                t1 = _mm_loadu_ps(temp2);
                for (j = k + 1; j + 4 < n; j += 4)
                {
                    t2 = _mm_loadu_ps(matrix[k] + j);
                    t3 = _mm_loadu_ps(matrix[i] + j);
                    t4 = _mm_mul_ps(t1, t2);
                    t3 = _mm_sub_ps(t3, t4);
                    _mm_storeu_ps(matrix[i] + j, t3);
                }
                for (j = j; j < n; j++)
                    matrix[i][j] -= matrix[i][k] * matrix[k][j];
                matrix[i][k] = 0;
            }
        }
        // 所有线程准备进入下一轮
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread_dynamic 并行算法
void calculate_pthread_dynamic()
{
    // 信号量初始化
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, THREAD_NUM);
    task=PTHREAD_MUTEX_INITIALIZER;

    // 创建线程
    pthread_t threads[THREAD_NUM];
    threadParam_t thread_param_t[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc_dynamic, (void *)(&thread_param_t[i]));
    }

    // 加入执行线程
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // 销毁信号量
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&task);
}

int main()
{
	int N=2000,count,cycle=1,step=10;
    float time = 0;
    init_data();
	for(n=10;n<=N;n+=step)
    {
        struct timeval start1,end1,start2,end2,start3,end3,start4,end4,start5,end5;
        if (n <= 30)
            cycle = 500;
        else if (n <= 100)
            cycle = 100;
        else if (n <= 300)
            cycle = 50;
        else cycle = 10;
        // ======================================normal ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start4, NULL);
            Gauss_Normal(n);
            gettimeofday(&end4, NULL);
            time += ((end4.tv_sec - start4.tv_sec) * 1000000 + (end4.tv_usec - start4.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "NORMAL："<<n<<time/cycle << "ms" << endl;
        // ======================================discrete ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start5, NULL);
            Gauss_SSE();
            gettimeofday(&end5, NULL);
            time += ((end5.tv_sec - start5.tv_sec) * 1000000 + (end5.tv_usec - start5.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "SSE："<<n<< time/cycle << "ms" << endl;
        // ======================================discrete ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start1, NULL);
            calculate_pthread_discrete();
            gettimeofday(&end1, NULL);
            time += ((end1.tv_sec - start1.tv_sec) * 1000000 + (end1.tv_usec - start1.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "discrete："<<n<< time/cycle << "ms" << endl;
        
        // ====================================== pthread ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start2, NULL);
            calculate_pthread_continuous();
            gettimeofday(&end2, NULL);
            time += ((end2.tv_sec - start2.tv_sec) * 1000000 + (end2.tv_usec - start2.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "continue："<<n<< time/ cycle << "ms" << endl;

        //====================================== avx ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start3, NULL);
            calculate_pthread_dynamic();
            gettimeofday(&end3, NULL);
            time += ((end3.tv_sec - start3.tv_sec) * 1000000 + (end3.tv_usec - start3.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "dynamic："<<n<<  time/ cycle << "ms" << endl;


		if(n==100) step=100;
        if(n==1000) step=500;
        cout<<endl<<endl;
	}
    return 0;
}

