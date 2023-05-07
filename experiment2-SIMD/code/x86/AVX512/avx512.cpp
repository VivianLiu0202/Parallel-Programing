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
using namespace std;
float Data[2000][2000],matrix[2000][2000];
int n;
//线程数据结构定义
typedef struct{
    int t_id;
}threadParam_t2;
const int num_threads=7;
//信号量定义
sem_t sem_leader;
sem_t sem_Division[num_threads-1];
sem_t sem_Elimination[num_threads-1];

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

void Gauss_AVX()
{
    int i,j,k;
    __m256_u t1,t2,t3,t4;//定义4个向量寄存器
    for(k = 0;k<n;k++)
    {
        float temp[8] = {matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
        t1 = _mm256_loadu_ps(temp);//加载到t1向量寄存器
        for(int j = k+1;j+8<=n;j+=8){
            t2=_mm256_loadu_ps(matrix[k]+j);//把内存中从b[k][j]开始的四个单精度浮点数加载到t2寄存器
            t3=_mm256_div_ps(t2,t1);//相除结果放到t3寄存器
            _mm256_storeu_ps(matrix[k]+j,t3);//把t3寄存器的值放回内存
        }
        for(j;j<n;j++){
            matrix[k][j]/=matrix[k][k];
        }
        matrix[k][k]=1.0;

        for(i=k+1;i<n;i++)
        {
            float temp2[8] = {matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
            t1 = _mm256_loadu_ps(temp2);
            for(j=k+1;j+8<=n;j+=8)
            {
                t2=_mm256_loadu_ps(matrix[k]+j);
                t3=_mm256_loadu_ps(matrix[k]+j);
                t4=_mm256_mul_ps(t1,t2);
                t3=_mm256_sub_ps(t3,t4);
                _mm256_storeu_ps(matrix[i]+j,t3);
            }
            for(j=j;j<n;j++) matrix[i][j] -= matrix[i][k]*matrix[k][j];
            matrix[i][k]=0;
        }
    }
}

void Gauss_AVX512()
{
    int i,j,k;
    __m512_u t1,t2,t3,t4;//定义4个向量寄存器
    for(k = 0;k<n;k++)
    {
        float temp[16]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
        t1 = _mm512_loadu_ps(temp);//加载到t1向量寄存器
        for(int j = k+1;j+16<=n;j+=16){
            t2=_mm512_loadu_ps(matrix[k]+j);//把内存中从b[k][j]开始的四个单精度浮点数加载到t2寄存器
            t3=_mm512_div_ps(t2,t1);//相除结果放到t3寄存器
            _mm512_storeu_ps(matrix[k]+j,t3);//把t3寄存器的值放回内存
        }
        for(j;j<n;j++){
            matrix[k][j]/=matrix[k][k];
        }
        matrix[k][k]=1.0;

        for(i=k+1;i<n;i++)
        {
            float temp2[16]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1 = _mm512_loadu_ps(temp2);
            for(j=k+1;j+16<=n;j+=16)
            {
                t2=_mm512_loadu_ps(matrix[k]+j);
                t3=_mm512_loadu_ps(matrix[k]+j);
                t4=_mm512_mul_ps(t1,t2);
                t3=_mm512_sub_ps(t3,t4);
                _mm512_storeu_ps(matrix[i]+j,t3);
            }
            for(j=j;j<n;j++) matrix[i][j] -= matrix[i][k]*matrix[k][j];
            matrix[i][k]=0;
        }
    }
}

void *threadFunc2(void* param)
{//SSE
    threadParam_t2 *p=(threadParam_t2*)param;
    int t_id=p->t_id; //线程编号
    for(int k=0;k<n;k++)
    {
        if(t_id==0)
        {
            float tmp=matrix[k][k];
            for(int j=k;j<n;j++)
                matrix[k][j]/=tmp;

        }
        else
        {
            sem_wait(&sem_Division[t_id-1]);
        }
		if(t_id==0)
        {
            for(int i=0;i<num_threads-1;i++)
                sem_post(&sem_Division[i]);
		}
		__m128 t1,t2,t3,t4; //定义4个向量寄存器
		for(int i=k+1+t_id;i<n;i+=num_threads)
		{
			float tmp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm_loadu_ps(tmp2);
            int j;
            for(j=k+1;j+4<=n;j+=4)
            {
                t2=_mm_loadu_ps(matrix[k]+j);
                t3=_mm_loadu_ps(matrix[i]+j);
                t4=_mm_mul_ps(t1,t2);
                t3=_mm_sub_ps(t3,t4);
                _mm_storeu_ps(matrix[i]+j,t3);
            }
            for(j=j;j<n;j++)
                matrix[i][j]-=matrix[i][k]*matrix[k][j];
            matrix[i][k]=0;
		}
		if(t_id==0){
            for(int i=0;i<num_threads-1;i++)
                sem_wait(&sem_leader);
            for(int i=0;i<num_threads-1;i++)
                sem_post(&sem_Elimination[i]);
        }
        else{
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id-1]);
        }
	}
	pthread_exit(NULL);
}

void *threadFunc3(void* param)
{//AVX
    threadParam_t2 *p=(threadParam_t2*)param;
    int t_id=p->t_id; //线程编号
    for(int k=0;k<n;k++)
    {
        if(t_id==0)
        {
            float tmp=matrix[k][k];
            for(int j=k;j<n;j++)
                matrix[k][j]/=tmp;

        }
        else
        {
            sem_wait(&sem_Division[t_id-1]);
        }
		if(t_id==0)
        {
            for(int i=0;i<num_threads-1;i++)
                sem_post(&sem_Division[i]);
		}
		__m256_u t1,t2,t3,t4; //定义4个向量寄存器
		for(int i=k+1+t_id;i<n;i+=num_threads)
		{
			float tmp2[8]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm256_loadu_ps(tmp2);
            int j;
            for(j=k+1;j+8<=n;j+=8)
            {
                t2=_mm256_loadu_ps(matrix[k]+j);
                t3=_mm256_loadu_ps(matrix[i]+j);
                t4=_mm256_mul_ps(t1,t2);
                t3=_mm256_sub_ps(t3,t4);
                _mm256_storeu_ps(matrix[i]+j,t3);
            }
            for(j=j;j<n;j++)
                matrix[i][j]-=matrix[i][k]*matrix[k][j];
            matrix[i][k]=0;
		}
		if(t_id==0)
        {
            for(int i=0;i<num_threads-1;i++)
                sem_wait(&sem_leader);
            for(int i=0;i<num_threads-1;i++)
                sem_post(&sem_Elimination[i]);
        }
        else
        {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id-1]);
        }
	}
	pthread_exit(NULL);
}


void *threadFunc4(void* param)
{//AVX512
    threadParam_t2 *p=(threadParam_t2*)param;
    int t_id=p->t_id; //线程编号
    for(int k=0;k<n;k++)
    {
        if(t_id==0)
        {
            float tmp=matrix[k][k];
            for(int j=k;j<n;j++)
                matrix[k][j]/=tmp;
        }
        else
        {
            sem_wait(&sem_Division[t_id-1]);
        }
		if(t_id==0)
        {
            for(int i=0;i<num_threads-1;i++)
                sem_post(&sem_Division[i]);
		}
		__m512_u t1,t2,t3,t4; //定义4个向量寄存器
		for(int i=k+1+t_id;i<n;i+=num_threads)
		{
		    int j;
            float temp2[16]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1 = _mm512_loadu_ps(temp2);
            for(j=k+1;j+16<=n;j+=16)
            {
                t2=_mm512_loadu_ps(matrix[k]+j);
                t3=_mm512_loadu_ps(matrix[k]+j);
                t4=_mm512_mul_ps(t1,t2);
                t3=_mm512_sub_ps(t3,t4);
                _mm512_storeu_ps(matrix[i]+j,t3);
            }
            for(j=j;j<n;j++) matrix[i][j] -= matrix[i][k]*matrix[k][j];
            matrix[i][k]=0;
        }
		if(t_id==0){
            for(int i=0;i<num_threads-1;i++)
                sem_wait(&sem_leader);
            for(int i=0;i<num_threads-1;i++)
                sem_post(&sem_Elimination[i]);
        }
        else{
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id-1]);
        }
	}
	pthread_exit(NULL);
}

void Gauss_pthreadSSE()
{
	int i;
	sem_init(&sem_leader,0,0);
	for(i=0;i<num_threads-1;i++)
    {
        sem_init(&sem_Division[i],0,0);
        sem_init(&sem_Elimination[i],0,0);
	}
	pthread_t *handles=new pthread_t[num_threads];//创建对应的handle
    threadParam_t2 *param=new threadParam_t2[num_threads];//创建对应的线程数据结构
    for(int t_id=0;t_id<num_threads;t_id++)
    {
        param[t_id].t_id=t_id;
    }
    for(int t_id=0;t_id<num_threads;t_id++)
    {
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

void Gauss_pthreadAVX()
{
    int i;
	sem_init(&sem_leader,0,0);
	for(i=0;i<num_threads-1;i++)
    {
        sem_init(&sem_Division[i],0,0);
        sem_init(&sem_Elimination[i],0,0);
	}
	pthread_t *handles=new pthread_t[num_threads];//创建对应的handle
    threadParam_t2 *param=new threadParam_t2[num_threads];//创建对应的线程数据结构
    for(int t_id=0;t_id<num_threads;t_id++)
    {
        param[t_id].t_id=t_id;
    }
    for(int t_id=0;t_id<num_threads;t_id++)
    {
        pthread_create(&handles[t_id],NULL,threadFunc3,&param[t_id]);
    }
    for(int t_id=0;t_id<num_threads;t_id++)
    {
        pthread_join(handles[t_id],NULL);
    }
	sem_destroy(sem_Division);
	sem_destroy(sem_Elimination);
	sem_destroy(&sem_leader);
}

void Gauss_pthreadAVX512()
{
    int i;
	sem_init(&sem_leader,0,0);
	for(i=0;i<num_threads-1;i++)
    {
        sem_init(&sem_Division[i],0,0);
        sem_init(&sem_Elimination[i],0,0);
	}
	pthread_t *handles=new pthread_t[num_threads];//创建对应的handle
    threadParam_t2 *param=new threadParam_t2[num_threads];//创建对应的线程数据结构
    for(int t_id=0;t_id<num_threads;t_id++)
    {
        param[t_id].t_id=t_id;
    }
    for(int t_id=0;t_id<num_threads;t_id++)
    {
        pthread_create(&handles[t_id],NULL,threadFunc4,&param[t_id]);
    }
    for(int t_id=0;t_id<num_threads;t_id++)
    {
        pthread_join(handles[t_id],NULL);
    }
	sem_destroy(sem_Division);
	sem_destroy(sem_Elimination);
	sem_destroy(&sem_leader);
}

void print_matrix()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main()
{
	int N=2000,count,cycle=1,step=10;
    float time = 0;
    init_data();
	for(n=10;n<=N;n+=step)
    {
        struct timeval start1,end1,start2,end2,start3,end3;
        struct timeval start4,end4,start5,end5,start6,end6;
        if (n <= 30)
            cycle = 500;
        else if (n <= 100)
            cycle = 100;
        else if (n <= 300)
            cycle = 50;
        else cycle = 10;
        // ====================================== sse ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start1, NULL);
            Gauss_SSE();
            gettimeofday(&end1, NULL);
            time += ((end1.tv_sec - start1.tv_sec) * 1000000 + (end1.tv_usec - start1.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "SSE："<<n<<" 时间："<< time/cycle << "ms" << endl;

        // ====================================== pthread ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start2, NULL);
            Gauss_pthreadSSE();
            gettimeofday(&end2, NULL);
            time += ((end2.tv_sec - start2.tv_sec) * 1000000 + (end2.tv_usec - start2.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "SSE pthread："<<n<<" 时间："<< time/ cycle << "ms" << endl;

        //====================================== avx ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start3, NULL);
            Gauss_AVX();
            gettimeofday(&end3, NULL);
            time += ((end3.tv_sec - start3.tv_sec) * 1000000 + (end3.tv_usec - start3.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "AVX："<<n<<" 时间："<<  time/ cycle << "ms" << endl;

        //====================================== pthread ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start4, NULL);
            Gauss_pthreadAVX();
            gettimeofday(&end4, NULL);
            time += ((end4.tv_sec - start4.tv_sec) * 1000000 + (end4.tv_usec - start4.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "AVX pthread："<<n<<" 时间："<<  time/ cycle << "ms" << endl;

        //====================================== pthread ======================================
        count=1;
           time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start5, NULL);
            Gauss_AVX512();
            gettimeofday(&end5, NULL);
            time += ((end5.tv_sec - start5.tv_sec) * 1000000 + (end5.tv_usec - start5.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "AVX512："<<n<<" 时间："<<  time/ cycle << "ms" << endl;

        //====================================== pthread ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start6, NULL);
            Gauss_pthreadAVX512();
            gettimeofday(&end6, NULL);
            time += ((end6.tv_sec - start6.tv_sec) * 1000000 + (end6.tv_usec - start6.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "AVX512 pthread："<<n<<" 时间："<<  time/ cycle << "ms" << endl;

		if(n==100) step=100;
        if(n==1000) step=500;
        cout<<endl<<endl;
	}
    return 0;
}



