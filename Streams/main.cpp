
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuControl.H>
#include <AMReX_BackgroundStream.H>

#include <thread>
#include <future>

#ifndef AMREX_USE_CUDA
#error This test is for CUDA only.
#endif

// Mocking primary thread work
void cpu_func (char name, int* count, double sleep, double time_zero)
{
    (*count)++;
    amrex::Print() << "Beginning " << name << " work, #" << *count << " at "
                   << (amrex::second() - time_zero)*1000.0 << " ms" << std::endl;
    amrex::Sleep(sleep);
    amrex::Print() << "Ending " << name << " work, #" << *count << " at "
                   << (amrex::second() - time_zero)*1000.0 << " ms" << std::endl;
}

// Mocking primary thread work
void gpu_func (char name, int* count, double sleep_ns, amrex::gpuStream_t stream)
{
   int nblocks = 1;
   int nthreads = 1;

    amrex::launch(nblocks, nthreads, stream,
    [=] AMREX_GPU_DEVICE () {
        (*count)++;
        AMREX_DEVICE_PRINTF("Starting Kernel %c, #%i \n", name, *count);
        __nanosleep(sleep_ns);
        AMREX_DEVICE_PRINTF("Ending Kernel %c, #%i \n", name, *count);
    });
}

// ......................

int main(int argc, char* argv[])
{
    using BackgroundStream = amrex::BackgroundStream;
    using gpuStream_t = amrex::gpuStream_t;

    amrex::Initialize(argc,argv);
    {
        double time_zero = amrex::second();

        int* hptr;
        CUdeviceptr dptr;
        AMREX_CUDA_SAFE_CALL(cudaHostAlloc((void**) &hptr, 3*sizeof(int), cudaHostAllocMapped));
        CU_CHECK(cuMemHostGetDevicePointer(&dptr, (void*) hptr, 0));
        hptr[0] = 0; hptr[1] = 0; hptr[2] = 0;

        amrex::Vector<BackgroundStream> streams(3);

        // CPU submit testing

        amrex::Print() << " Starting cpus " << amrex::second() - time_zero << std::endl;

        streams[0].cpuSubmit( [=]() { cpu_func(char('A'), hptr+0, 0.01, time_zero); } );
        streams[1].cpuSubmit( [=]() { cpu_func(char('B'), hptr+1, 0.01, time_zero); } );
        streams[2].cpuSubmit( [=]() { cpu_func(char('C'), hptr+2, 0.01, time_zero); } );
        streams[0].cpuSubmit( [=]() { cpu_func(char('D'), hptr+0, 0.01, time_zero); } );
        streams[1].cpuSubmit( [=]() { cpu_func(char('E'), hptr+1, 0.01, time_zero); } );
        streams[2].cpuSubmit( [=]() { cpu_func(char('F'), hptr+2, 0.01, time_zero); } );

        amrex::Print() << " Waiting for cpus to finish at: " << amrex::second() - time_zero << std::endl;
        streams[0].cpuSync();
        streams[1].cpuSync();
        streams[2].cpuSync();
        amrex::Print() << "  done at " << amrex::second() - time_zero << std::endl;

        // ==================================
        // GPU submit testing

        amrex::Print() << std::endl << std::endl;
        amrex::Print() << " Starting gpus: " << amrex::second() - time_zero << std::endl;

        streams[0].gpuSubmit( [=](gpuStream_t s) { gpu_func(char('a'), (int*) dptr+0, 100, s); } );
        streams[1].gpuSubmit( [=](gpuStream_t s) { gpu_func(char('b'), (int*) dptr+1, 10000, s); } );
        streams[1].gpuSubmit( [=](gpuStream_t s) { gpu_func(char('c'), (int*) dptr+1, 10000, s); } );
        streams[1].gpuSubmit( [=](gpuStream_t s) { gpu_func(char('d'), (int*) dptr+1, 10000, s); } );
        streams[1].gpuSubmit( [=](gpuStream_t s) { gpu_func(char('e'), (int*) dptr+1, 10000, s); } );
        streams[2].gpuSubmit( [=](gpuStream_t s) { gpu_func(char('f'), (int*) dptr+2, 100, s); } );

        amrex::Print() << " Waiting for gpus to finish at: " << amrex::second() - time_zero << std::endl;
        streams[0].gpuSync();
        streams[1].gpuSync();
        streams[2].gpuSync();
        amrex::Print() << "  done at " << amrex::second() - time_zero << std::endl;

        amrex::Sleep(1.0);

        // ==================================
        // Ping-pong testing

        amrex::Print() << std::endl << std::endl;
        amrex::Print() << " Ping-pong testing starting at " << amrex::second() - time_zero << std::endl;

        streams[0].gpuSubmit( [=](gpuStream_t s) { gpu_func(char('1'), (int*) dptr+0, 10000, s); } );
        streams[1].gpuSubmit( [=](gpuStream_t s) { gpu_func(char('A'), (int*) dptr+1, 10000, s); } );
        streams[0].cpuSubmit( [=]()              { cpu_func(char('2'), hptr+0, 0.0001, time_zero); } );
        streams[1].cpuSubmit( [=]()              { cpu_func(char('B'), hptr+1, 0.0001, time_zero); } );
        streams[0].gpuSubmit( [=](gpuStream_t s) { gpu_func(char('3'), (int*) dptr+0, 20000, s); } );
        streams[1].gpuSubmit( [=](gpuStream_t s) { gpu_func(char('C'), (int*) dptr+1, 20000, s); } );
        streams[0].cpuSubmit( [=]()              { cpu_func(char('4'), hptr+0, 0.0002, time_zero); } );
        streams[1].cpuSubmit( [=]()              { cpu_func(char('D'), hptr+1, 0.0002, time_zero); } );
        streams[0].gpuSubmit( [=](gpuStream_t s) { gpu_func(char('5'), (int*) dptr+0, 30000, s); } );
        streams[1].gpuSubmit( [=](gpuStream_t s) { gpu_func(char('E'), (int*) dptr+1, 30000, s); } );
        streams[0].cpuSubmit( [=]()              { cpu_func(char('6'), hptr+0, 0.0003, time_zero); } );
        streams[1].cpuSubmit( [=]()              { cpu_func(char('F'), hptr+1, 0.0003, time_zero); } );

        amrex::Print() << " Waiting for gpus to finish at: " << amrex::second() - time_zero << std::endl;
        streams[0].sync();
        streams[1].sync();
        streams[2].sync();
        amrex::Print() << "  done at " << amrex::second() - time_zero << std::endl;

        AMREX_CUDA_SAFE_CALL(cudaFreeHost((void*) hptr));

        // ==================================
        // Dependency testing
        // Simple sequence w/ all combos

        amrex::Print() << std::endl << std::endl;
        amrex::Print() << " Dependency testing starting at " << amrex::second() - time_zero << std::endl;

        streams[0].cpuSubmit              ( [=]()              { cpu_func(char('1'), hptr+0, 0.0001, time_zero); });
        streams[1].gpuSubmitwithDependency( [=](gpuStream_t s) { gpu_func(char('a'), (int*) dptr+0, 10000, s); }, streams[0] );
        streams[1].gpuSubmit              ( [=](gpuStream_t s) { gpu_func(char('b'), (int*) dptr+0, 10000, s); });
        streams[0].cpuSubmitwithDependency( [=]()              { cpu_func(char('2'), hptr+0, 0.0001, time_zero); }, streams[1]);
        streams[0].cpuSubmit              ( [=]()              { cpu_func(char('3'), hptr+0, 0.0001, time_zero); });
        streams[1].cpuSubmitwithDependency( [=]()              { cpu_func(char('4'), hptr+0, 0.0001, time_zero); }, streams[0]);
        streams[1].gpuSubmit              ( [=](gpuStream_t s) { gpu_func(char('c'), (int*) dptr+0, 10000, s); });
        streams[0].gpuSubmitwithDependency( [=](gpuStream_t s) { gpu_func(char('d'), (int*) dptr+0, 10000, s); }, streams[1] );

        amrex::Print() << " Attempt 2 " << std::endl;

        streams[0].cpuSubmitwithDependency( [=]()              { cpu_func(char('1'), hptr+0, 0.0001, time_zero); }, streams[0]);
        streams[1].gpuSubmitwithDependency( [=](gpuStream_t s) { gpu_func(char('a'), (int*) dptr+0, 10000, s); }, streams[0] );
        streams[1].gpuSubmitwithDependency( [=](gpuStream_t s) { gpu_func(char('b'), (int*) dptr+0, 10000, s); }, streams[1]);
        streams[0].cpuSubmitwithDependency( [=]()              { cpu_func(char('2'), hptr+0, 0.0001, time_zero); }, streams[1]);
        streams[0].cpuSubmitwithDependency( [=]()              { cpu_func(char('3'), hptr+0, 0.0001, time_zero); }, streams[0]);
        streams[1].cpuSubmitwithDependency( [=]()              { cpu_func(char('4'), hptr+0, 0.0001, time_zero); }, streams[0]);
        streams[1].gpuSubmitwithDependency( [=](gpuStream_t s) { gpu_func(char('c'), (int*) dptr+0, 10000, s); }, streams[1]);
        streams[0].gpuSubmitwithDependency( [=](gpuStream_t s) { gpu_func(char('d'), (int*) dptr+0, 10000, s); }, streams[1] );

        amrex::Print() << " Waiting for dependencies to finish at: " << amrex::second() - time_zero << std::endl;
        streams[0].sync();
        streams[1].sync();
        amrex::Print() << "  done at " << amrex::second() - time_zero << std::endl;

        // ==================================

    }
    amrex::Finalize();

    return 0;
}
