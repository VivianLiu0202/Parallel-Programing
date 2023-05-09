#include<iostream>
#include<sys/time.h>
#include<pthread.h>
#include<stdlib.h>
#include<semaphore.h>
#include<arm_neon.h>
using namespace std;
float Data[2000][2000],matrix[2000][2000];
int n;
//线程数据结构定义
typedef struct{
    int t_id;
}threadParam_t2;
const int num_threads=20;
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

void normal()
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

void NEON()
{
    int i,j,k;
    float32x4_t t1,t2,t3,t4; //定义4个向量寄存器
    for(k=0;k<n;k++)
    {
        float32x4_t t1=vmovq_n_f32(matrix[k][k]);
        j = k+1; 
        while((k*n+j)%4!=0)
        {//对齐操作
            matrix[k][j]=matrix[k][j]*1.0/matrix[k][k];
            j++;
        }
        for(;j+4<=n;j+=4)
        {
            t2=vld1q_f32(matrix[k]+j); //把内存中从B[k][j]开始的四个单精度浮点数加载到t2寄存器
            t3=vdivq_f32(t2,t1); //相除结果放到t3寄存器
            vst1q_f32(matrix[k]+j,t3); //把t3寄存器的值放回内存
        }
        for(j;j<n;j++) //处理剩下的不能被4整除的
            matrix[k][j]/=matrix[k][k];
        matrix[k][k]=1.0;
        //以上完成了对第一个部分的向量化

        for(i=k+1;i<n;i++)
        {
            float32x4_t t1 =  vmovq_n_f32(matrix[i][k]);
            j=k+1;
            while((k*n+j)%4!=0){//对齐操作
                matrix[i][j]=matrix[i][j]-matrix[k][j]*matrix[i][k];
                j++;
            }
            for(;j+4<=n;j+=4)
            {
                t2=vld1q_f32(matrix[k]+j);
                t3=vld1q_f32(matrix[i]+j);
                t4=vmulq_f32(t1,t2);
                t3=vsubq_f32(t3,t4);
                vst1q_f32(matrix[i]+j,t3);
            }
            for(j=j;j<n;j++) matrix[i][j] -= matrix[i][k]*matrix[k][j];
            matrix[i][k]=0.0;
        }
    }
}

void *threadFunc2(void* param){//循环划分
    threadParam_t2 *p=(threadParam_t2*)param;
    int t_id=p->t_id; //线程编号
    for(int k=0;k<n;k++)
    {
        if(t_id==0)
        {
            float32x4_t Akk = vmovq_n_f32(matrix[k][k]);
            int j;
            for (j = k + 1; j + 3 < n; j += 4)
            {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                Akj = vdivq_f32(Akj, Akk);
                vst1q_f32(matrix[k] + j, Akj);
            }
            for (; j < n; j++)
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
		for(int i=k+1+t_id;i<n;i+=num_threads)
		{
			float32x4_t t1 = vmovq_n_f32(matrix[i][k]);
            for(j=k+1;j+4<=n;j+=4)
            {
                t2=vld1q_f32(matrix[k]+j);
                t3=vld1q_f32(matrix[i]+j);
                t4=vmulq_f32(t1,t2);
                t3=vsubq_f32(t3,t4);
                vst1q_f32(matrix[i]+j,t3);
            }
            for(j=j;j<n;j++)
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

void Neon_pthread(){
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
    threadParam_t2 *param=new threadParam_t2[num_threads];//创建对应的线程数据结构
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
	int N=2000,count,cycle=10,step=10;
    float time = 0;
    init_data();
    n=1000;
	// for(n=10;n<=N;n+=step)
    // {
        struct timeval start1,end1,start2,end2,start3,end3;
        // if (n <= 30)
        //     cycle = 500;
        // else if (n <= 100)
        //     cycle = 100;
        // else if (n <= 300)
        //     cycle = 50;
        // else cycle = 10;
        // ====================================== serial ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start1, NULL);
            normal();
            gettimeofday(&end1, NULL);
            time += ((end1.tv_sec - start1.tv_sec) * 1000000 + (end1.tv_usec - start1.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "普通："<<n<<" 时间："<< time/cycle << "ms" << endl;
        
        // ====================================== neon ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start2, NULL);
            NEON();
            gettimeofday(&end2, NULL);
            time += ((end2.tv_sec - start2.tv_sec) * 1000000 + (end2.tv_usec - start2.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "NEON："<<n<<" 时间："<< time/ cycle << "ms" << endl;

        // ====================================== pthread ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start3, NULL);
            Neon_pthread();
            gettimeofday(&end3, NULL);
            time += ((end3.tv_sec - start3.tv_sec) * 1000000 + (end3.tv_usec - start3.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "NEON的pthread版本："<<n<<" 时间："<<  time/ cycle << "ms" << endl;

		// if(n==100) step=100;
        // if(n==1000) step=500;
        // cout<<endl<<endl;
	// }
    return 0;
}
