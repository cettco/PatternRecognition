#include"Bp.h"
double m[innode];         //test input
void readit()
{
    FILE *fp;
    if((fp=fopen("r.txt","r"))==NULL)
    {   
        cout<<"open file failed"<<endl;
        exit(0);
    }
    for(int i = 0;i<innode;i++)
    {
        fscanf(fp,"%lf",&m[i]);
    }
    fclose(fp);
}
int main()
{
    readit();
    BpNet bp = BpNet();
    bp.init(); 
    bp.readtrain();
    double *r=bp.recognize(m); 
    for(int i = 0;i<outnode;i++)
    {
        cout<<"result:"<<bp.result[i]<<endl;
    }
    double max = 0;
    int index;
    double temp;
    for(int i = 0;i<outnode;i++)
    {
        temp = (double)fabs(bp.result[i]);
        if(temp>max)
        {
            max = temp;
            index = i;
        }
    }
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
    cout<<endl;
    cout<<"The result is :"<<index<<endl;
}