#!/bin/bash

# echo "Hello, GeeksforGeeks"
# for ((domain_size_x=256;domain_size_x<=512;domain_size_x=domain_size_x*2))
#     do
#     for ((domain_size_y=256;domain_size_y<=512;domain_size_y=domain_size_y*2))
#         do
#         for ((domain_size_z=256;domain_size_z<=512;domain_size_z=domain_size_z*2))
#             do
#             for ((nbins=2;nbins<=6000;nbins=nbins*2))
#                 do

#                 # nbins=2
#                 # domain_size=256
#                 max_grid_size=$((domain_size_x/nbins))
#                 grid_size=$((max_grid_size/2))
                
#                 # boxes=8
#                 for ((i=max_grid_size; i>=grid_size && grid_size > 0; i=i/2))
#                 do
#                     for ((j=i;  j>=grid_size; j=j/2))
#                     do
#                         for((k=j; k>=grid_size; k=k/2))
#                         do
#                             echo "i=$i,j=$j,k=$k"
#                             boxes=$(($((domain_size/i))*$((domain_size/j))*$((domain_size/k))))
#                             echo "nbins=$nbins"
#                             echo "boxes=$boxes"
#                             echo "max_grid_size=$max_grid_size"
#                             echo  "grid_size=$grid_size"
#                             cmd='./main3d.gnu.x86-milan.TPROF.ex inputs  nbins='$nbins' domain="('$domain_size_x,$domain_size_y,$domain_size_z')" max_grid_size="('$i,$j,$k')" >output/'$domain_size'_'$nbins'_'$boxes'_output.txt'
#                             echo $cmd
#                             # eval $cmd
#                         done
                        
#                     done
#                 done
#             done
#         done
#     done
# done
# ./main3d.gnu.x86-milan.TPROF.OMP.ex inputs 
# ./main3d.gnu.x86-milan.TPROF.ex inputs 
# ./main3d.gnu.x86-milan.TPROF.ex inputs  nnodes=128 max_grid_size="(32,16,16)" >output_withfixedseed/128_2048_output.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs  nbins=2 max_grid_size="(128,128,64)" >output/2_16_combined_output_output.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs  nbins=2 max_grid_size="(128,64,64)" >output/2_32_output.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs  nbins=2 max_grid_size="(64,64,64)" >output/2_64_combined_output.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs  nbins=2 max_grid_size="(64,64,32)" >output/2_128_combined_output.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs  nbins=2 max_grid_size="(64,32,32)" >output/2_256_combined_output.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs  nbins=2 max_grid_size="(32,32,32)" >output/2_512_combined_output.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs  nbins=2 max_grid_size="(32,32,16)" >output/2_1024_output.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs  nbins=2 max_grid_size="(32,16,16)" >output/2_2048_output.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs  nbins=2 max_grid_size="(16,16,16)" >output/2_4096_output.txt



# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=2 domain="(256,256,256)" max_grid_size="(128,128,128)" >output/2_4_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=2 domain="(256,256,256)" max_grid_size="(128,128,64)" >output/2_8_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=2 domain="(256,256,256)" max_grid_size="(128,64,64)" >output/2_16_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=4 domain="(256,256,256)" max_grid_size="(128,128,64)" >output/4_4_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=4 domain="(256,256,256)" max_grid_size="(128,64,64)" >output/4_8_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=4 domain="(256,256,256)" max_grid_size="(64,64,64)" >output/4_16_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=8 domain="(256,256,256)" max_grid_size="(64,64,64)" >output/8_4_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=8 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/8_8_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=8 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/8_16_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=16 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/16_4_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=16 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/16_8_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=16 domain="(256,256,256)" max_grid_size="(32,32,32)" >output/16_16_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=32 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/32_4_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=32 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/32_8_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=32 domain="(256,256,256)" max_grid_size="(32,32,32)" >output/32_16_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=64 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/64_4_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=64 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/64_8_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=64 domain="(256,256,256)" max_grid_size="(32,32,32)" >output/64_16_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=128 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/128_4_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=128 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/128_8_output_best.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=128 domain="(256,256,256)" max_grid_size="(32,32,32)" >output/128_16_output_best.txt


# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=2 domain="(256,256,256)" max_grid_size="(128,128,128)" >output/2_4_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=2 domain="(256,256,256)" max_grid_size="(128,128,64)" >output/2_8_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=2 domain="(256,256,256)" max_grid_size="(128,64,64)" >output/2_16_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=4 domain="(256,256,256)" max_grid_size="(128,128,64)" >output/4_4_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=4 domain="(256,256,256)" max_grid_size="(128,64,64)" >output/4_8_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=4 domain="(256,256,256)" max_grid_size="(64,64,64)" >output/4_16_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=8 domain="(256,256,256)" max_grid_size="(64,64,64)" >output/8_4_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=8 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/8_8_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=8 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/8_16_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=16 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/16_4_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=16 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/16_8_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=16 domain="(256,256,256)" max_grid_size="(32,32,32)" >output/16_16_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=32 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/32_4_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=32 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/32_8_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=32 domain="(256,256,256)" max_grid_size="(32,32,32)" >output/32_16_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=64 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/64_4_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=64 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/64_8_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=64 domain="(256,256,256)" max_grid_size="(32,32,32)" >output/64_16_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=128 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/128_4_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=128 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/128_8_output_avg.txt
# ./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=128 domain="(256,256,256)" max_grid_size="(32,32,32)" >output/128_16_output_avg.txt


./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=2 domain="(256,256,256)" max_grid_size="(128,128,128)" >output/2_4_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=2 domain="(256,256,256)" max_grid_size="(128,128,64)" >output/2_8_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=2 domain="(256,256,256)" max_grid_size="(128,64,64)" >output/2_16_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=4 domain="(256,256,256)" max_grid_size="(128,128,64)" >output/4_4_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=4 domain="(256,256,256)" max_grid_size="(128,64,64)" >output/4_8_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=4 domain="(256,256,256)" max_grid_size="(64,64,64)" >output/4_16_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=8 domain="(256,256,256)" max_grid_size="(64,64,64)" >output/8_4_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=8 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/8_8_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=8 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/8_16_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=16 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/16_4_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=16 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/16_8_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=16 domain="(256,256,256)" max_grid_size="(32,32,32)" >output/16_16_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=32 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/32_4_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=32 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/32_8_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=32 domain="(256,256,256)" max_grid_size="(32,32,32)" >output/32_16_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=64 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/64_4_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=64 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/64_8_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=64 domain="(256,256,256)" max_grid_size="(32,32,32)" >output/64_16_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=128 domain="(256,256,256)" max_grid_size="(64,64,32)" >output/128_4_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=128 domain="(256,256,256)" max_grid_size="(64,32,32)" >output/128_8_output_worst.txt
./main3d.gnu.x86-milan.TPROF.ex inputs nnodes=128 domain="(256,256,256)" max_grid_size="(32,32,32)" >output/128_16_output_worst.txt