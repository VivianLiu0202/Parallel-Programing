#include<iostream>
#include<sys/time.h>
#include<pthread.h>
#include<stdlib.h>
#include<semaphore.h>
using namespace std;
float Data[2000][2000],matrix[2000][2000];
int n;
//线程数据结构定义
typedef struct{//静态
    int t_id;
}threadParam_t2;
const int num_threads=7;

//信号量定义
typedef struct{//动态
    int k;
    int t_id;
}threadParam_t;

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

void *threadFunc(void* param)
{//动态线程版本
    threadParam_t *p=(threadParam_t*)param;
    int k=p->k; //消去轮次
    int t_id=p->t_id; //线程编号
    int i=k+t_id+1; //计算任务起始点
    for(i;i<n;i+=num_threads)
    {
        float tmp2=matrix[i][k];
        for(int j=k+1;j<n;j++)
            matrix[i][j]-=tmp2*matrix[k][j];
        matrix[i][k]=0;
    }
    pthread_exit(NULL);
}

void *threadFunc2(void* param){//静态线程+信号量同步版本
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
        else{
            sem_wait(&sem_Division[t_id-1]);
        }
		if(t_id==0)
        {
            for(int i=0;i<num_threads-1;i++)
                sem_post(&sem_Division[i]);
		}
		for(int i=k+1+t_id;i<n;i+=num_threads)
		{
			float tmp2=matrix[i][k];
			for(int j=k+1;j<n;j++)
            {
				matrix[i][j]-=tmp2*matrix[k][j];
            }
			matrix[i][k]=0.0;
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

void Gauss_para_active()
{
	int i,j,k;
	for(k=0;k<n;k++)
    {
        //主线程进行乘法操作
		float tmp=matrix[k][k];
		for(j=k+1;j<n;j++)
			matrix[k][j]/=tmp;
        matrix[k][k]=1.0;

        //创建工作线程进行消去工作
        int work_count = 7;//n-1-k;
        //所以为什么这里动态线程耗时特别久
        
        pthread_t* handles=new pthread_t[work_count];//创建对应的handle
        threadParam_t* param=new threadParam_t[work_count];//创建对应的线程数据结构
        //分配任务，也即初始化参数
        for(int t_id=0;t_id<work_count;t_id++)
        {
            param[t_id].k=k;
            param[t_id].t_id=t_id;
        }
        //创建线程
        for(int t_id=0;t_id<work_count;t_id++)
        {
            pthread_create(&handles[t_id],NULL,threadFunc,&param[t_id]);
        }
        //主线程挂起等待所有的工作线程完成此轮消去工作
        for(int t_id=0;t_id<work_count;t_id++)
        {
            pthread_join(handles[t_id],NULL);
        }

	}
}
void Gauss_para_static()
{
	int i;
	sem_init(&sem_leader,0,0);
	for(i=0;i<num_threads-1;i++){
        sem_init(&sem_Division[i],0,0);
        sem_init(&sem_Elimination[i],0,0);
	}
	pthread_t *handles=new pthread_t[num_threads];//创建对应的handle
    threadParam_t2 *param=new threadParam_t2[num_threads];//创建对应的线程数据结构
    for(int t_id=0;t_id<num_threads;t_id++)
    {
        param[t_id].t_id=t_id;
    // }
    // for(int t_id=0;t_id<num_threads;t_id++){
        pthread_create(&handles[t_id],NULL,threadFunc2,&param[t_id]);
    }
    for(int t_id=0;t_id<num_threads;t_id++){
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
        if (n <= 30)
            cycle = 500;
        else if (n <= 100)
            cycle = 100;
        else if (n <= 300)
            cycle = 50;
        else cycle = 10;
        // ====================================== 动态 ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start1, NULL);
            Gauss_para_active();
            gettimeofday(&end1, NULL);
            time += ((end1.tv_sec - start1.tv_sec) * 1000000 + (end1.tv_usec - start1.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "动态线程："<<n<<" 时间："<< time/cycle << "ms" << endl;
        
        // ====================================== 静态 ======================================
        count=1;
        time=0;
		while(count<cycle)
        {
            init_matrix();
            gettimeofday(&start2, NULL);
            Gauss_para_static();
            gettimeofday(&end2, NULL);
            time += ((end2.tv_sec - start2.tv_sec) * 1000000 + (end2.tv_usec - start2.tv_usec)) * 1.0 / 1000;
            count++;
        }
		cout << "静态线程："<<n<<" 时间："<< time/ cycle << "ms" << endl;

		if(n==100) step=100;
        if(n==1000) step=500;
        cout<<endl<<endl;
	}
	return 0;
}
