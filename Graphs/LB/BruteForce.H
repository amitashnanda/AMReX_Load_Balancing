#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring>
//#include "mpi.h"
using namespace std;
//bool array_true=false;

char* getfn(int it, char* b, char* c, char* d);

double get_maxt(int *rank, long int *guess, int N,int nr);

void BruteForceDoIt(int nbins, int nitems, double mean, double stdev,long int *guess);

int* ternary(int n, int nr,int N);
    
void basechange_fill(int N, int nr,long int *guess);


void get_all_combos(int N, int nr, long int *guess, int *ranks);
