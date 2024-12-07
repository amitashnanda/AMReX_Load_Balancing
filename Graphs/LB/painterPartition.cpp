#include <AMReX_Extension.H>
#include <AMReX_IntVect.H>
#include <AMReX_Morton.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelContext.H>

// A DP based CPP program for painter's partition problem 
#include <climits> 
#include <iostream> 
#include <vector>
#include <SFC.H>
#include <Knapsack.H>
#include <LeastUsed.H>
using namespace std; 
// struct SFCToken
// {
//     class Compare
//     {
//     public:
//         AMREX_FORCE_INLINE
//         bool operator () (const SFCToken& lhs,
//                           const SFCToken& rhs) const;
//     };
//     int m_box;
//     amrex::Array<uint32_t,AMREX_SPACEDIM> m_morton;
// };
// namespace {

//     AMREX_FORCE_INLINE
//     SFCToken makeSFCToken (int box_index, amrex::IntVect const& iv)
//     {
//         SFCToken token;
//         token.m_box = box_index;

// #if (AMREX_SPACEDIM == 3)

//         constexpr int imin = -(1 << 29);
//         AMREX_ASSERT_WITH_MESSAGE(AMREX_D_TERM(iv[0] >= imin && iv[0] < -imin,
//                                             && iv[1] >= imin && iv[1] < -imin,
//                                             && iv[2] >= imin && iv[2] < -imin),
//                                   "SFCToken: index out of range");
//         uint32_t x = iv[0] - imin;
//         uint32_t y = iv[1] - imin;
//         uint32_t z = iv[2] - imin;
//         // extract lowest 10 bits and make space for interleaving
//         token.m_morton[0] = amrex::Morton::makeSpace(x & 0x3FF)
//                          | (amrex::Morton::makeSpace(y & 0x3FF) << 1)
//                          | (amrex::Morton::makeSpace(z & 0x3FF) << 2);
//         x = x >> 10;
//         y = y >> 10;
//         z = z >> 10;
//         token.m_morton[1] = amrex::Morton::makeSpace(x & 0x3FF)
//                          | (amrex::Morton::makeSpace(y & 0x3FF) << 1)
//                          | (amrex::Morton::makeSpace(z & 0x3FF) << 2);
//         x = x >> 10;
//         y = y >> 10;
//         z = z >> 10;
//         token.m_morton[2] = amrex::Morton::makeSpace(x & 0x3FF)
//                          | (amrex::Morton::makeSpace(y & 0x3FF) << 1)
//                          | (amrex::Morton::makeSpace(z & 0x3FF) << 2);

// #elif (AMREX_SPACEDIM == 2)

//         constexpr uint32_t offset = 1u << 31;
//         static_assert(static_cast<uint32_t>(std::numeric_limits<int>::max())+1 == offset,
//                       "INT_MAX != (1<<31)-1");
//         uint32_t x = (iv[0] >= 0) ? static_cast<uint32_t>(iv[0]) + offset
//             : static_cast<uint32_t>(iv[0]-std::numeric_limits<int>::lowest());
//         uint32_t y = (iv[1] >= 0) ? static_cast<uint32_t>(iv[1]) + offset
//             : static_cast<uint32_t>(iv[1]-std::numeric_limits<int>::lowest());
//         // extract lowest 16 bits and make sapce for interleaving
//         token.m_morton[0] = amrex::Morton::makeSpace(x & 0xFFFF)
//                          | (amrex::Morton::makeSpace(y & 0xFFFF) << 1);
//         x = x >> 16;
//         y = y >> 16;
//         token.m_morton[1] = amrex::Morton::makeSpace(x) | (amrex::Morton::makeSpace(y) << 1);

// #elif (AMREX_SPACEDIM == 1)

//         constexpr uint32_t offset = 1u << 31;
//         static_assert(static_cast<uint32_t>(std::numeric_limits<int>::max())+1 == offset,
//                       "INT_MAX != (1<<31)-1");
//         token.m_morton[0] = (iv[0] >= 0) ? static_cast<uint32_t>(iv[0]) + offset
//             : static_cast<uint32_t>(iv[0]-std::numeric_limits<int>::lowest());

// #else
//         static_assert(false,"AMREX_SPACEDIM != 1, 2 or 3");
// #endif

//         return token;
//     }
// }
// function to calculate sum between two indices 
// in wgts
long  sum(vector<long> wgts, int from, int to) 
{ 
	long int total = 0; 
	for (int i = from; i <= to; i++) 
		total += wgts[i]; 
	
	return total; 
} 

// bottom up tabular dp 
long  findMax(vector<long> wgts, int n, int k) 
{ 
	// initialize table 
	long int dp[k + 1][n + 1] = { 0 }; 

	// base cases 
	// k=1 
	for (int i = 1; i <= n; i++) 
		dp[1][i] = sum(wgts, 0, i - 1); 

	// n=1 
	for (int i = 1; i <= k; i++) 
		dp[i][1] = wgts[0]; 

	// 2 to k partitions 
	for (int i = 2; i <= k; i++) { // 2 to n boards 
		for (int j = 2; j <= n; j++) { 

			// track minimum 
			long  best = LONG_MAX; 
			

			// i-1 th separator before position wgts[p=1..j] 
			for (int p = 1; p <= j; p++) 
				best = min(best, max(dp[i - 1][p], 
							sum(wgts, p, j - 1)));	 

			dp[i][j] = best; 
		} 
	} 
	
	 
	return dp[k][n]; 
} 
bool isPartitionPossible(vector<long> wgts, int n, int k, long maxTime){
        
        long long currTime=0, worker=1;
        for(int i=0;i<n;i++)
        {
            if(currTime+wgts[i]>maxTime){
                worker++;
                
                if(worker>k) return false;
                currTime=wgts[i];
            }
            else currTime+=wgts[i];
        }
        return true;
        
    }
long minWeight(vector<long> wgts, int n, int k)
    {
        // code here
        // return minimum time
        long long sum=0,max=wgts[0];
       // if(k>n) return -1;
        for(int i=0;i<n;i++){
            sum+=wgts[i];
            if(max<wgts[i]){
                max=wgts[i];
            }
        }
        long long h=sum,l=max,mid,res;
        while(l<=h){
            mid=l+(h-l)/2;
            if(isPartitionPossible(wgts,n,k,mid)){
                res=mid;
                h=mid-1;
            }
            else{
                l=mid+1;
            }
        }
        return res;
        
    }

// driver function 
 vector<int> painterPartition(const amrex::BoxArray&   boxes,vector<long> wgts,int k) 
{ 

	BL_PROFILE("painterPartition()");
    vector<long> sorted_wgts;
	
	const int N = boxes.size();
    std::vector<SFCToken> tokens;
    tokens.reserve(N);
    for (int i = 0; i < N; ++i)
    {
        const amrex::Box& bx = boxes[i];
        tokens.push_back(makeSFCToken(i, bx.smallEnd()));
    }

	std::sort(tokens.begin(), tokens.end(), SFCToken::Compare());
	for (int i = 0; i < N; ++i)
		{
			// amrex::Print() << tokens[i].m_box << " \n";
            sorted_wgts.push_back(wgts[tokens[i].m_box]);
            // amrex::Print() << tokens[i].m_morton[0] << " " << tokens[i].m_morton[1] << " " << tokens[i].m_morton[2] << " \n";
        }

	long int n=wgts.size();

    // for (int i = 0; i < N; ++i)
	// 	{
	// 		amrex::Print() << wgts[i] <<" " <<sorted_wgts[i]<<" \n";
    //     }
	//int k = 2; 
	long maxVal=minWeight(sorted_wgts,n,k);
	//cout << maxVal << endl; 
	vector< vector<int> > vec(k);
    vector<int> sorted_result(wgts.size());
	int index;
	
    
	amrex::Real  s_painter_eff=0.0;
	bool sort=true;
	int nteams=k;
	bool  flag_verbose_mapper=true;
	for(int i=0,j=i, p=0;p<k && j<n;p++){
		
		long val=sum(sorted_wgts,i,j);
		while(maxVal>val && j<n){
			j++;
			val=sum(sorted_wgts,i,j);
		}
		// cout<<"i= "<<i<<" j= "<<j<<endl;
		if(maxVal==val){
			for(int a=i;a<=j;a++){
				index=a;
				vec[p].push_back(index);
                //cout<<"p="<<p<<endl;
                sorted_result[index]=p;
            }
            j++;
			i=j;
		}
		else{
			for(int a=i;a<j;a++){
				index=a;
				vec[p].push_back(index);
               // cout<<"p="<<p<<endl;
                sorted_result[index]=p;
			}
			i=j;
		}

	}
   
   vector<int> result(wgts.size());
   for (int i = 0; i < N; ++i)
		{
			// amrex::Print() << tokens[i].m_box << " \n";
            //sorted_wgts.push_back(wgts[tokens[i].m_box]);
            result[tokens[i].m_box] = sorted_result[i];
            // amrex::Print() << tokens[i].m_morton[0] << " " << tokens[i].m_morton[1] << " " << tokens[i].m_morton[2] << " \n";
        }
    // amrex::Print()<<"----------------result\n";
    // for (int j = 0; j < N; ++j) {
    //     amrex::Print()<<wgts[j]<<" , "<<result[j]<<" ";
    
    // }
    // amrex::Print()<<"----------------\n";
    // for (int j = 0; j < N; ++j) {
       
    //     amrex::Print()<<sorted_wgts[j]<<" , "<<sorted_result[j]<<" ";
    // }
	// for(int i=0;i<k;i++){
    //     for(int j=0;j<vec[i].size();j++){
	// 		cout<<"vec["<<i<<"]["<<j<<"]="<<vec[i][j]<<endl;
	// 	}
	// 	cout<<endl;
	// }

	std::vector<LIpair> LIpairV;

    LIpairV.reserve(nteams);
    
    for (int i = 0; i < nteams; ++i)
    {
        amrex::Long wgt = 0;
        const std::vector<int>& vi = vec[i];
        for (int j = 0, M = vi.size(); j < M; ++j)
            {   
                // amrex::Print()<<"vi["<<j<<"]="<<vi[j]<<endl;
                wgt += sorted_wgts[vi[j]];}

        LIpairV.push_back(LIpair(wgt,i));
    }
	if (sort) Sort(LIpairV, true);

    // if (flag_verbose_mapper) {
    //     for (const auto &p : LIpairV) {
    //         amrex::Print() << "  Bucket " << p.second << " contains " << p.first << std::endl;
    //     }
    // }

	if (s_painter_eff || flag_verbose_mapper)
    {
        amrex::Real sum_wgt = 0, max_wgt = 0;
        for (int i = 0; i < nteams; ++i)
        {
            const amrex::Long W = LIpairV[i].first;
            if (W > max_wgt) max_wgt = W;
            sum_wgt += W;
        }
        amrex::Real efficiency = (sum_wgt/(nteams*max_wgt));
        if (s_painter_eff) s_painter_eff = efficiency;

        if (flag_verbose_mapper)
        {

            //amrex::Print()<<__LINE__<<std::endl;
            amrex::Print() << "SFC+painterPartition efficiency: " << efficiency << '\n';
        }
    }
     
	return result;
} 