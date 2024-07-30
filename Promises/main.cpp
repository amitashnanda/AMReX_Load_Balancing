
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_Gpu.H>

#include <thread>
#include <future>

#ifndef AMREX_USE_CUDA
#error This test is for CUDA only.
#endif

struct gpu_promise {
    char id;
    double sleep;
    double tz;
    std::promise<void> timestep;
};

struct gpu_basic {
    char id;
    double sleep;
    double tz;
};

// ......................

// Mocking GPU callback function work
void CUDART_CB do_step_gpu_trigger(void* p)
{
    auto p_prom = reinterpret_cast<gpu_promise*>(p);

    amrex::Print() << "Beginning GPU work on " << p_prom->id <<
                      " at " << amrex::second() - p_prom->tz << std::endl;
    amrex::Sleep(p_prom->sleep);
    amrex::Print() << "Ending GPU work on " << p_prom->id <<
                      " at " << amrex::second() - p_prom->tz << std::endl;

    p_prom->timestep.set_value();
    delete p_prom;
}

// Mocking GPU callback function work
void CUDART_CB do_step_gpu(void* p)
{
    auto p_basic = reinterpret_cast<gpu_basic*>(p);

    amrex::Print() << "Beginning GPU work on " << p_basic->id <<
                      " at " << amrex::second() - p_basic->tz << std::endl;
    amrex::Sleep(p_basic->sleep);
    amrex::Print() << "Ending GPU work at " << p_basic->id <<
                      " at " << amrex::second() - p_basic->tz << std::endl;

    delete p_basic;
}

// ......................

// Mocking primary thread work
void do_step (char id, double sleep, double time_zero)
{
    amrex::Print() << "Beginning host work on " << id << " at " << amrex::second() - time_zero << std::endl;
    amrex::Sleep(sleep);
    amrex::Print() << "Ending host work on " << id << " at " << amrex::second() - time_zero << std::endl;
}

// ......................

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        amrex::Print() << "Hello world from AMReX version " << amrex::Version() << "\n";

        int max_step = 3;
        int timestep = 0;

        double sleep = 0.1;
        double sleep_thread = 0.5;
        double time_zero = amrex::second();

        const int max_streams = amrex::Gpu::Device::numGpuStreams();
        std::vector<amrex::gpuStream_t> streams(max_streams);
        for (int i=0; i<max_streams; ++i)
        {
            amrex::Gpu::Device::setStreamIndex(i);
            streams[i] = amrex::Gpu::gpuStream();
        }

        for (; timestep < max_step; timestep++)
        {
            std::promise<void> timestep_promise;
            std::future<void> timestep_future = timestep_promise.get_future();

            std::promise<void> finish_promise;
            std::future<void> finish_future = finish_promise.get_future();

            amrex::Print() << "======================================" << std::endl;
            amrex::Print() << "Starting timestep " << timestep
                           << " at " << amrex::second() - time_zero << std::endl;

            auto p = new gpu_promise({'A', sleep_thread, time_zero, std::move(timestep_promise)});
            auto b = new gpu_basic  ({'B', sleep_thread, time_zero});
            auto f = new gpu_promise({'C', sleep_thread, time_zero, std::move(finish_promise)});
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(streams[0], do_step_gpu_trigger, (void*) p));
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(streams[0], do_step_gpu, (void*) b));
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(streams[0], do_step_gpu_trigger, (void*) f));

            do_step('a', sleep, time_zero);

            amrex::Print() << "Waiting for GPU work in " << timestep
                           << " to finish at " << amrex::second() - time_zero << std::endl;

            timestep_future.wait();

            do_step('b', sleep, time_zero);

            finish_future.wait();

            amrex::Print() << "Finishing timestep " << timestep
                           << " at " << amrex::second() - time_zero << std::endl;

        }

        amrex::Print() << "======================================" << std::endl;
    }
    amrex::Finalize();

    return 0;
}
