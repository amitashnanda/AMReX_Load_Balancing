#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring>
#include <cmath>
// #include <boost/random/beta_distribution.hpp>
// #include <boost/random/mersenne_twister.hpp>
using namespace std;
bool array_true=false;

char* getfn(int it, char* b, char* c, char* d, const char* dist)
{
    char * fn = new char[30 + std::strlen(b) + std::strlen(c) + std::strlen(dist) + 5];
    if (it == 1) std::strcpy(fn, "guess_"); 
    else if (it == 2) std::strcpy(fn, "ranks_");
    else if (it == 3) std::strcpy(fn, "combo_");
    else if (it == 4) std::strcpy(fn, "norep_");

    std::strcat(fn, b);
    std::strcat(fn, "_");
    std::strcat(fn, c);
    std::strcat(fn, "_");
    std::strcat(fn, dist);
    std::strcat(fn, "_it");
    std::strcat(fn, d);

    if((it == 3) || (it == 4)){
        int world_rank = 0;
        char r[6 + sizeof(char)];
        std::sprintf(r, "%d", world_rank);    
        std::strcat(fn, ".");
        std::strcat(fn, r);
    }
    std::strcat(fn, ".txt");
    return fn;
}

double get_maxt(int *rank, double *guess, int N,int nr) {
    double maxg[nr]={0};
    double maxt=0;
    for (int r=0;r<nr;r++){ 
        for (int i=0;i<N;i++) { 
            if(rank[i]==r){
                maxg[r]+=guess[i]; 
            }
        }
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
    
void basechange_fill(int N, int nr,double *guess,ofstream& myfile) {
  
    double nmin=0;
    double nmax=pow(nr,(N));//16
    
    int i=nmin;
    while((i>=nmin) && (i<nmax)){  
        int * combo = new int[N+1];
        for (int j=0;j<N;j++){
            combo[j] = 0;
        }
       
        if (nr!=1){
            combo=ternary(i,nr,N); 
            

            if (combo[0]==-1) {	    
                cout<<"Nmax too large"<<endl;
		        exit(0);
	        }
        }
        double maxt=get_maxt(combo,guess,N,nr);
        for (int j=0;j<N;j++){
	    myfile<<combo[j]<<" ";
	    }
	    myfile<<maxt<<endl;
	    i++;
	}
}


void get_all_combos(int N, int nr, double *guess, int *ranks, char *fn){

    
    int combo[N];
    int ln=0;

    ofstream myfile;
    myfile.open(fn);
    basechange_fill(N, nr,guess,myfile);
    myfile.close();
    return;
    }

void countFreq(int arr[], int n, int *ranks, int *freqs)
{
   
    vector<bool> visited(n, false);
 
    
    for (int i = 0; i < n; i++) {
 
        
        if (visited[i] == true)
            continue;
 
        
        int count = 1;
        for (int j = i + 1; j < n; j++) {
            if (arr[i] == arr[j]) {
                visited[j] = true;
                count++;
            }
        }
        ranks[i]=arr[i];
        freqs[i]=count;
    }
}

void eliminate_repetitions(int N, int nr, char *fn, char *tfn){
    
    ifstream myfile;
    myfile.open(fn);
    ofstream myfile2;
    myfile2.open(tfn);

    std::string line;
    while(std::getline(myfile,line))
    {
	bool write=true;
        int * combo = new int[N];
	std::string r;
	double guess;
	int j=0;
	int z=0;
        for (int i=0;i<line.length();i++)
	{
	    if((line[i]!=' ')&&(i<line.length()-1))
	    {
                if(z==0){
		    r=line[i];
		}
		else{
                    r+=line[i];
		}
		z++;
	    }
	    else
	    { //we have reached a space or the end of the line, store what we read and reset r
                if(j<N and i>0)
		{
                    combo[j]=stoi(r);
		    z=0;
	            j++;
		}
		if(j==N)
		{
                    guess=stof(r);
		}
	    }
	}
      

	int * ranks = new int[N];
	int * freqs = new int[N];
	countFreq(combo,N,ranks,freqs);
	for (int i=0;i<N;i++)
	{
            for (int j=0;j<N;j++)
		{
		if((freqs[i]==freqs[j])&(i>j)&(ranks[i]<ranks[j]))
			write=false;
		}
	}
	if(write)	
	{
	    myfile2<<line<<endl;
	}
	
	//cout<< guess<<endl;//guess is at lower precision than originally, should be to 6sf		   
    }
    myfile2.close();
    myfile.close();
}


int main(int argc, char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now();

    char *itn = new char[10];
    if (argc != 6 && argc != 7){
        cout << "usage: ./a.out Nranks Nbox_per_rank mean stddev distribution" << endl;
        return 0;
    } 
    else {
        int nr = atoi(argv[1]);
        int N = atoi(argv[2]) * nr;
        int mean = atoi(argv[3]); 
        int stddev = atoi(argv[4]);
        std::string distribution_type = argv[5];
        if(argc == 7){
            itn = argv[6];
        } 
        else {
            std::strcpy(itn, "0\0");
        }

        double guess[N];
        int ranks[N];

        char * guess_fn = getfn(1, argv[3], argv[4], itn,distribution_type.c_str());
        char * ranks_fn = getfn(2, argv[3], argv[4], itn,distribution_type.c_str());
        char * combo_fn = getfn(3, argv[3], argv[4], itn,distribution_type.c_str());
        char * temps_fn = getfn(4, argv[3], argv[4], itn,distribution_type.c_str());
       
        ofstream myfile;
        myfile.open(ranks_fn);
        for (int i = 0; i < nr; i++){
            ranks[i] = i;
            myfile << ranks[i] << " ";
        }
        myfile.close();

        std::default_random_engine generator;
        if (distribution_type == "normal") {
            std::normal_distribution<double> distribution(mean, stddev);
            myfile.open(guess_fn);
            for (int i = 0; i < N; i++){
                guess[i] = distribution(generator);
                if (guess[i] < 0){
                    cout << "negative guess, make mean higher or standard deviation smaller" << endl;
                    exit(0);
                }
                myfile << guess[i] << " ";
            }
            myfile.close();
        } else if (distribution_type == "uniform") {
            std::uniform_real_distribution<double> distribution(mean - stddev, mean + stddev);
            myfile.open(guess_fn);
            for (int i = 0; i < N; i++){
                guess[i] = distribution(generator);
                myfile << guess[i] << " ";
            }
            myfile.close();
        } else if (distribution_type == "exponential") {
            std::exponential_distribution<double> distribution(1.0 / mean);
            myfile.open(guess_fn);
            for (int i = 0; i < N; i++){
                guess[i] = distribution(generator);
                myfile << guess[i] << " ";
            }
            myfile.close();
        } else if (distribution_type == "lognormal") {
            std::lognormal_distribution<double> distribution(mean, stddev);
            myfile.open(guess_fn);
            for (int i = 0; i < N; i++){
                guess[i] = distribution(generator);
                myfile << guess[i] << " ";
            }
            myfile.close();
        } else {
            cout << "Unknown distribution type. Supported  distribution types are: normal, uniform, exponential, lognormal." << endl;
            exit(0);
        }

        get_all_combos(N, nr, guess, ranks, combo_fn);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        cout << "Time taken: " << elapsed_time.count() << " seconds" << endl;

        return 0;
    }
}