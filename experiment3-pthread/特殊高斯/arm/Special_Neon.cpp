#include <iostream>
#include <fstream>
#include <sstream>
#include<bitset>
#include<arm_neon.h>
#include<sys/time.h>
using namespace std;
const int Columnnum = 254;
const int Rnum = 106;
const int Enum = 53;
const int ArrayColumn = 8;//8 = 254 / 32 ceiling
const int leftbit = 2;//32 - (1 + 253 % 32) = 2
unsigned int R [Columnnum][ArrayColumn];
unsigned int E [Enum][ArrayColumn];
int First[Enum];
bitset<32> MyBit(0);
uint32x4_t vaR;
uint32x4_t vaE;
int Find_First(int index)
{
    int j = 0;
    int cnt = 0;
    while(E[index][j] == 0)
    {
        j++;
        if(j == ArrayColumn) break;
    }
    if(j == ArrayColumn) return -1;
    unsigned int tmp = E[index][j];
    while(tmp != 0){
        tmp = tmp >> 1;
        cnt++;
    }
    return Columnnum - 1 - ((j+1)*32 - cnt - leftbit);
}
void Init_Eliminator(){
    unsigned int a;
    ifstream infile("Eliminator.txt");
    char fin[5000] = {0};
    int index;
    while(infile.getline(fin,sizeof(fin)))
    {
        std::stringstream line(fin);
        bool flag = 0;
        while(line >> a)
        {
            if(flag == 0)
            {
                index = a;
                flag = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            R[index][ArrayColumn - 1 - j] += temp;
        }
    }
}
void Init_EliminatedLine()
{
    unsigned int a;
    ifstream infile("EliminatedLine.txt");
    char fin[5000] = {0};
    int index = 0;
    while(infile.getline(fin,sizeof(fin)))
    {
        std::stringstream line(fin);
        int flag = 0;
        while(line >> a){
            if(flag == 0)
            {
                First[index] = a;
                flag = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            E[index][ArrayColumn - 1 - j] += temp;
        }
        index++;
    }
}
bool Is_NULL(int index)
{
    for(int j = 0;j < ArrayColumn; j++)
    {
        if(R[index][j] != 0) return 0;
    }
    return 1;
}
void Set_R(int eindex,int rindex)
{
    for(int j = 0;j < ArrayColumn; j++){

        R[rindex][j] = E[eindex][j];
    }
}
void XOR(int eindex,int rindex){//we do parallel programming here.
    int j = 0;
    for(;j + 4 <= ArrayColumn; j += 4)
    {
        vaE = vld1q_u32(&(E[eindex][j]));
        vaR = vld1q_u32(&(R[rindex][j]));
        vaE = veorq_u32(vaE,vaR);
        vst1q_u32(&(E[eindex][j]),vaE);
    }
    for(;j < ArrayColumn; j++)
    {
        E[eindex][j] = E[eindex][j] ^ R[rindex][j];
    }

}
void Align_XOR(int eindex,int rindex){//we do parallel programming here.
    int j = 0;
    while((eindex * ArrayColumn + j) % 4 != 0){//we do alignment here.
        E[eindex][j] = E[eindex][j] ^ R[rindex][j];
        j++;
    }
    for(;j + 4 <= ArrayColumn; j += 4){
        vaE = vld1q_u32(&(E[eindex][j]));
        vaR = vld1q_u32(&(R[rindex][j]));
        vaE = veorq_u32(vaE,vaR);
        vst1q_u32(&(E[eindex][j]),vaE);
    }
    for(;j < ArrayColumn; j++){
        E[eindex][j] = E[eindex][j] ^ R[rindex][j];
    }

}
void Neon(){
    for(int i = 0;i < Enum; i++){
        while(First[i] != -1){
            if(!Is_NULL(First[i])){
                XOR(i,First[i]);
                //Align_XOR(i,First[i]);
                First[i] = Find_First(i);
            }
            else{
                Set_R(i,First[i]);
                break;
            }
        }
    }
}
void Print(){//Print the answer
    for(int i = 0;i < Enum; i++){
        if(First[i] == -1){
            cout<<endl;
            continue;
        }
        for(int j = 0;j < ArrayColumn; j++){
            if(E[i][j] == 0) continue;
            MyBit = E[i][j];//MENTION: bitset manipulates from the reverse direction
            for(int k = 31;k >= 0; k--){
                if(MyBit.test(k)){
                    cout<<32 * (ArrayColumn - j - 1) + k<<' ';
                }
            }
        }
        cout<<endl;
    }
}
int main()
{

    struct timeval head;
    struct timeval tail;
    Init_Eliminator();
    Init_EliminatedLine();
    gettimeofday(&head,NULL);
    Neon();
    gettimeofday(&tail,NULL);
    cout<<"Special Gauss, Neon version, Enum: "<<Enum<<", Time: "<<(tail.tv_sec-head.tv_sec)*1000.0+(tail.tv_usec-head.tv_usec)/1000.0<<"ms"<<endl;
    return 0;
}