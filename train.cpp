#include"Bp.h"
double m[innode];         //test input
// void readit()
// {
//     FILE *fp;
//     if((fp=fopen("r.txt","r"))==NULL)
//     {   
//         cout<<"open file failed"<<endl;
//         exit(0);
//     }
//     for(int i = 0;i<innode;i++)
//     {
//         fscanf(fp,"%lf",&m[i]);
//     }
//     fclose(fp);
// }
//输入样本  

int main()  
{  
    cout<<"hello"<<endl;
    //initData(); 
    BpNet bp;  
    bp.init(); 
    bp.initData();
    int times=0;  
    while(bp.error>0.0001)  
    {  
        bp.e=0.0;  
        times++;  
        bp.train(bp.X,bp.Y);  
        cout<<"Times="<<times<<" error="<<bp.error<<endl;  
    }  
    bp.writetrain();
    cout<<"trainning complete..."<<endl;   
    // double *r=bp.recognize(m);  
    // for(int i = 0;i<outnode;i++)
    // {
    //     cout<<bp.result[i]<<endl;
    // }
    // double mi =100;
    // int index;
    // double cha[trainsample][outnode];
    // for(int i = 0;i<trainsample;i++)
    // {
    //     int j;
    //     for(j = 0;j<outnode;j++)
    //     {
    //         cha[i][j]+=(double)fabs(bp.Y[i][j]-bp.result[j]);
    //     }
    //     if(cha[i][j]<mi)
    //     {
    //         mi = cha[i][j];
    //         index = i;
    //     }
    // }
    // cout<<endl;
    // cout<<"The result is :"<<index<<endl;
    // for(int i=0;i<outnode;++i)  
    //    cout<<bp.result[i]<<" ";  
    // double cha[trainsample][outnode];  
    // double mi=100;  
    // double index;  
    // for(int i=0;i<trainsample;i++)  
    // {  
    //     for(int j=0;j<outnode;j++)  
    //     {  
    //         //找差值最小的那个样本  
    //         cha[i][j]=(double)(fabs(Y[i][j]-bp.result[j]));  
    //         if(cha[i][j]<mi)  
    //         {  
    //             mi=cha[i][j];  
    //             index=i;  
    //         }  
    //     }  
    // }  
    // for(int i=0;i<innode;++i)  
    //    cout<<m[i];  
    // cout<<" is "<<index<<endl;  
    // cout<<endl;  
    return 0;  
    }