#include <AMReX.H>
#include <AMReX_Random.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>


#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring>
#include <BruteForce.H>
#include <AMReX_Print.H>
#include <omp.h>
//#include "mpi.h"
using namespace std;
bool array_true=false;

char* getfn(int it, char* b, char* c, char* d)
{
    char * fn = new char[20+std::strlen(b)+strlen(c)+5];
    if (it == 1)std::strcpy(fn,"guess_"); 
    else if(it == 2)std::strcpy(fn,"ranks_");

    else if(it == 3)std::strcpy(fn,"combo_");
    else if(it == 4)std::strcpy(fn,"norep_");

    std::strcat(fn,b);
    std::strcat(fn,"_");
    std::strcat(fn,c);
    std::strcat(fn,"_it");
    std::strcat(fn,d);
    
   

    if((it==3)||(it==4)){
        int world_rank=0;
       // MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	char r [6+sizeof(char)];
        std::sprintf(r,"%d",world_rank);	
	std::strcat(fn,".");
	std::strcat(fn,r);
    }
     std::strcat(fn,".txt");
    return fn;
}

double get_maxt(int *rank, long int *guess, int N,int nr) {
    double maxg[nr]={0};
    double maxt=0;
    
    for (int r=0;r<nr;r++){ // for every bucket
        #pragma omp parallel for
        for (int i=0;i<N;i++) { //for every element
            if(rank[i]==r){
                maxg[r]+=guess[i]; // this guess is on bucket r so add to the sum
            }
        }
        #pragma omp critical
        if (maxg[r]>maxt){
            maxt=maxg[r];
        }
    }
    return maxt;
}

int* ternary(int n, int nr,int N) {
    int * nums = new int[N];
    for(int i=0;i<N;i++) nums[i]=0;
    int L=N;
    //for  the call is ternary(0,2,4)
    //n=8,nr=2, N=4
    //nums[]=0 0 0 0 
    //for rank 1, the call is ternary(9,2,4)
    // n=9, nr=2, N=4
    //n=9/2=4
    //r=1, nums[4]=1
    // n=64/2=32
    //r=0, nums[3]=0
    //... nums=[-1 0 1 0 1]
    //n=6, nr=2, N=4
    //r=0, nums[]=1

    while (n){
        auto dv= std::div(n,nr);
        n = dv.quot;
	    int r = dv.rem;
        if (L==0){
                nums[0]=-1;
            break;
        }
        L--;
        nums[L]=r;
        }
        return nums;
}
    
void basechange_fill(int N, int nr,long int *guess) {
    //int world_size,world_rank;

    //MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    //double nmin=double(world_rank)/double(world_size)*pow(nr,(N));
    //double nmax=(world_rank+1.0)/double(world_size)*pow(nr,(N));
    // nmin for 
    /*
    suppose  nr=number of buckets=2, Nbox_per_bucket=2, N=2*2=4
    for 0 rank
    nmin=0/2*pow(2,4)=0
    nmax=pow(2,4)=16
    
    */
    double nmin=0;
    double nmax=pow(nr,(N));//16
    amrex::Print()<<"N= "<<N<<" nr= "<<nr<<" nmax= "<<nmax<<endl;
    int i=nmin;
    // amrex::Print()<<__LINE__<<endl;
    while((i>=nmin) && (i<nmax)){  // for rank 0, i=0 
        int * combo = new int[N+1];
        #pragma omp parallel for
        for (int j=0;j<N;j++){
            combo[j] = 0;
        }
        // amrex::Print()<<__LINE__<<endl;
        //check there is more than one bucket
        if (nr!=1){
            #pragma omp parallel
            combo=ternary(i,nr,N);    // it will work with combinations of bucket picks, which bucket pick what.
            //for the call is ternary(0,2,4)
            //for the call is ternary(1,2,4)
            //for the call is ternary(3,2,4)
            //......
            //for the call is ternary(15,2,4)

            if (combo[0]==-1) {	    
                cout<<"Nmax too large"<<endl;
		        exit(0);
	        }
        }
        // amrex::Print()<<__LINE__<<endl;
        double maxt=get_maxt(combo,guess,N,nr);
        // for (int j=0;j<N;j++){
	    //    // myfile<<combo[j]<<" ";
	    //    amrex::Print()<<combo[j]<<" ";
	    //     }
	    // myfile<<maxt<<endl;
	    i++;
	}
}

void BruteForceDoIt(int nbins, int nitems, double mean, double stdev,long int *guess){

    BL_PROFILE("BruteForceDoIt()");
    int N=nbins*nitems;
    int nr=nbins;
    int ranks [N];

    // Parallel section to find local maxima and then the global maximum
    // #pragma omp parallel
    // {
    // amrex::Print()<< "Total number of threads: "<<omp_get_thread_num()<<endl;
    // }
    //double guess [N];
    
    // char *itn = new char[10];

	// std::strcpy(itn,"0\0");
    
    // std::string mean_str = std::to_string(mean);
    // char* mean_cstr = new char[mean_str.length() + 1];  // Allocate memory for the C-string
    // std::strcpy(mean_cstr, mean_str.c_str());

    // std::string stdev_str = std::to_string(stdev);
    // char* stdev_cstr = new char[stdev_str.length() + 1];  // Allocate memory for the C-string
    // std::strcpy(stdev_cstr, stdev_str.c_str());
    // // generate fiNames
    // char * guess_fn = getfn(1,mean_cstr,stdev_cstr,itn);
    // char * ranks_fn = getfn(2,mean_cstr,stdev_cstr,itn);
    // char * combo_fn = getfn(3,mean_cstr,stdev_cstr,itn);
    // char * temps_fn = getfn(4,mean_cstr,stdev_cstr,itn);
    
    // delete[] mean_cstr;
    // delete[] stdev_cstr;

    for (int i = 0; i<nr;i++){
        ranks[i]=i;
       // myfile << ranks[i] << " ";   // write list of buckets 
    }
    // for (int i=0;i<N;i++){
    //     //guess[i]=amrex::RandomNormal(mean, stdev); //guess value based on the normal distribution
	//     if (guess[i]<0){
    //         cout<<"negative guess, make mean higher or standard deviation smaller"<<endl;
    //             exit(0);
    //     }
    // }
    
    amrex::Print()<<__LINE__<<endl;
     get_all_combos(N,nr,guess,ranks);


}
void get_all_combos(int N, int nr, long int *guess, int *ranks){

    //double tot = double(pow(N,nr)); -- no use of this line
    // int combo[N];
    // int ln=0;
     
    // ofstream myfile;
    // myfile.open(fn);
    basechange_fill(N, nr,guess);
    // myfile.close();
    return;
    }

// void countFreq(int arr[], int n, int *ranks, int *freqs)
// {
//     // Mark all array elements as not visited
//     vector<bool> visited(n, false);
 
//     // Traverse through array elements and
//     // count frequencies
//     for (int i = 0; i < n; i++) {
 
//         // Skip this element if already processed
//         if (visited[i] == true)
//             continue;
 
//         // Count frequency
//         int count = 1;
//         for (int j = i + 1; j < n; j++) {
//             if (arr[i] == arr[j]) {
//                 visited[j] = true;
//                 count++;
//             }
//         }
//         ranks[i]=arr[i];
//         freqs[i]=count;
//     }
// }

// void eliminate_repetitions(int N, int nr, char *fn, char *tfn){
    
//     ifstream myfile;
//     myfile.open(fn);
//     ofstream myfile2;
//     myfile2.open(tfn);

//     std::string line;
//     while(std::getline(myfile,line))
//     {
// 	    bool write=true;
//         int * combo = new int[N];
// 	    std::string r;
// 	    double guess;
// 	    int j=0;
// 	    int z=0;
//         for (int i=0;i<line.length();i++)
// 	    {
// 	        if((line[i]!=' ')&&(i<line.length()-1))
// 	        {
//                  if(z==0){
// 		         r=line[i];
// 		    }
// 		    else{
//                  r+=line[i];
// 		    }
// 		    z++;
// 	        }
// 	        else
// 	         { //we have reached a space or the end of the line, store what we read and reset r
//                 if(j<N and i>0)
// 		        {
//                     combo[j]=stoi(r);
// 		            z=0;
// 	                j++;
// 		        }
//             if(j==N)
//             {
//                         guess=stof(r);
//             }
//             }
// 	}
      

// 	int * ranks = new int[N];
// 	int * freqs = new int[N];
// 	countFreq(combo,N,ranks,freqs);
// 	for (int i=0;i<N;i++)
// 	{
//             for (int j=0;j<N;j++)
// 		{
// 		if((freqs[i]==freqs[j])&(i>j)&(ranks[i]<ranks[j]))
// 			write=false;
// 		}
// 	}
// 	if(write)	
// 	{
// 	    myfile2<<line<<endl;
// 	}
	
// 	//cout<< guess<<endl;//guess is at lower precision than originally, should be to 6sf		   
//     }
//     myfile2.close();
//     myfile.close();
// }


// int main( int argc, char* argv[]) {


//     auto start_time = std::chrono::high_resolution_clock::now();

//     char *itn = new char[10];
//     if (argc!=5 and argc!=6){
//     cout<< "usage: ./a.out Nranks Nbox_per_rank mean stddev" <<endl;
//     return 0;
//     } 
//     else{

//     // get inputs 
//     int nr=atoi(argv[1]); //nr= number of bucket
//     int N=atoi(argv[2])*nr; //boxes per bucket is the new input. So, N is total number of box
//     int mean=atoi(argv[3]); 
//     int stddev=atoi(argv[4]);
//     if(argc==6){
//         itn=argv[5];
//     } 
//     else{
// 	    std::strcpy(itn,"0\0");
//     }
//     // initialize arrays
//     double guess [N];
//     int ranks [N];

//     //Initialize MPI
//     //MPI_Init(NULL,NULL);
//     // generate fiNames
//     char * guess_fn = getfn(1,argv[3],argv[4],itn);
//     char * ranks_fn = getfn(2,argv[3],argv[4],itn);
//     char * combo_fn = getfn(3,argv[3],argv[4],itn);
//     char * temps_fn = getfn(4,argv[3],argv[4],itn);
   
//     ofstream myfile;
//     myfile.open(ranks_fn);
//     for (int i = 0; i<nr;i++){
//         ranks[i]=i;
//         myfile << ranks[i] << " ";   // write list of buckets 
//     }
//     myfile.close();

//     //generate array of guesses with normal distribution
//     std::default_random_engine generator;
//     std::normal_distribution<double> distribution(mean,stddev);
//     myfile.open(guess_fn);
//     for (int i=0;i<N;i++){
//         guess[i]=distribution(generator);  //guess value based on the normal distribution
// 	if (guess[i]<0){
// 	    cout<<"negative guess, make mean higher or standard devation smaller"<<endl;
//             exit(0);
// 	}
// 	myfile << guess[i] << " ";  //write on the guess file guess_meanValue_stdValue_it0
//     }
//     myfile.close();
//     get_all_combos(N,nr,guess,ranks,combo_fn);  // N=total number of boxes, nr= number of buckets 
//                                                 //guess=based on normal distribution 
//                                                 //ranks= collection of ranks number suppose 0-63 for 64 ranks
// //   eliminate_repetitions(N,nr,combo_fn,temps_fn);
//    // MPI_Finalize();
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

//     cout << "Time taken: " << elapsed_time.count() << " seconds" << endl;

//     return 0;
// }
// }
