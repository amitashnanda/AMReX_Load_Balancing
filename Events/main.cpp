
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_Gpu.H>

#include <thread>
#include <future>

#ifndef AMREX_USE_GPU
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
void CUDART_CB do_step_gpu_promise(void* p)
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

// Mocking CPU thread work
void do_step_promise (double sleep, double time_zero, std::promise<void> timestep)
{
    amrex::Print() << "Beginning thread work at " << amrex::second() - time_zero << std::endl;
    amrex::Sleep(sleep);
    amrex::Print() << "Ending thread work at " << amrex::second() - time_zero << std::endl;

    timestep.set_value();
}

// Mocking primary thread work
void do_step (double sleep, double time_zero)
{
    amrex::Print() << "Beginning host work at " << amrex::second() - time_zero << std::endl;
    amrex::Sleep(sleep);
    amrex::Print() << "Ending host work at " << amrex::second() - time_zero << std::endl;
}

// ......................

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        amrex::Print() << "Hello world from AMReX version " << amrex::Version() << "\n";

        int max_step = 3;

        double sleep_ns = 0.5*1e9;
        double sleep_host = 0.1;
        double sleep_host_fast = 0.05;
        double time_zero = amrex::second();

        const int max_streams = amrex::Gpu::Device::numGpuStreams();
        std::vector<amrex::gpuStream_t> streams(max_streams);
        std::vector<cudaEvent_t> events(max_streams);
        for (int i=0; i<max_streams; ++i)
	{
            amrex::Gpu::Device::setStreamIndex(i);
            streams[i] = amrex::Gpu::gpuStream();
            cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
        }

        for (int timestep = 0; timestep < max_step; timestep++)
        {
            // Events 
            // ............................

            amrex::Print() << "======================================" << std::endl;
            amrex::Print() << "Starting events " << timestep
                           << " at " << amrex::second() - time_zero
                           << " with sleep of " << sleep_ns << std::endl;

            amrex::single_task(streams[0],
            [=] AMREX_GPU_DEVICE () {
                AMREX_DEVICE_PRINTF("Starting Sync Kernel A\n");
                __nanosleep(sleep_ns);
                AMREX_DEVICE_PRINTF("Ending Sync Kernel A\n");
            });

            amrex::single_task(streams[1],
            [=] AMREX_GPU_DEVICE () {
                AMREX_DEVICE_PRINTF("Starting Sync Kernel B\n");
                __nanosleep(sleep_ns);
                AMREX_DEVICE_PRINTF("Ending Sync Kernel B\n");
            });

            amrex::single_task(streams[2],
            [=] AMREX_GPU_DEVICE () {
                AMREX_DEVICE_PRINTF("Starting Sync Kernel C\n");
                __nanosleep(sleep_ns);
                AMREX_DEVICE_PRINTF("Ending Sync Kernel C\n");
            });

            amrex::single_task(streams[3],
            [=] AMREX_GPU_DEVICE () {
                AMREX_DEVICE_PRINTF("Starting Sync Kernel D\n");
                __nanosleep(sleep_ns);
                AMREX_DEVICE_PRINTF("Ending Sync Kernel D\n");
            });

            // ...................
            // Do this call back between the two sets of kernels.

            for (int i=0; i<max_streams; ++i) {
                cudaEventRecord(events[i], streams[i]);
            }

            for (int j=0; j<max_streams; ++j) {
                for (int i=0; i<max_streams; ++i) {
                    cudaStreamWaitEvent(streams[i], events[i], 0);
                }
            }

            auto b = new gpu_basic{'*', sleep_host_fast, time_zero};
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(streams[0], do_step_gpu, (void*) b));

            cudaEventRecord(events[0], streams[0]);

            for (int j=0; j<max_streams; ++j) {
                for (int i=0; i<max_streams; ++i) {
                    cudaStreamWaitEvent(streams[i], events[0], 0);
                }
            }

            // ...................

            amrex::single_task(streams[0],
            [=] AMREX_GPU_DEVICE () {
                AMREX_DEVICE_PRINTF("Starting Sync Kernel E\n");
                __nanosleep(sleep_ns);
                AMREX_DEVICE_PRINTF("Ending Sync Kernel E\n");
            });

            amrex::single_task(streams[1],
            [=] AMREX_GPU_DEVICE () {
                AMREX_DEVICE_PRINTF("Starting Sync Kernel F\n");
                __nanosleep(sleep_ns);
                AMREX_DEVICE_PRINTF("Ending Sync Kernel F\n");
            });

            amrex::single_task(streams[2],
            [=] AMREX_GPU_DEVICE () {
                AMREX_DEVICE_PRINTF("Starting Sync Kernel G\n");
                __nanosleep(sleep_ns);
                AMREX_DEVICE_PRINTF("Ending Sync Kernel G\n");
            });

            amrex::single_task(streams[3],
            [=] AMREX_GPU_DEVICE () {
                AMREX_DEVICE_PRINTF("Starting Sync Kernel H\n");
                __nanosleep(sleep_ns);
                AMREX_DEVICE_PRINTF("Ending Sync Kernel H\n");
            });

            do_step(sleep_host, time_zero);

            amrex::Gpu::synchronize();
            amrex::Print() << "Finishing events on " << timestep
                           << " at " << amrex::second() - time_zero << std::endl;
        }

        amrex::Print() << "======================================" << std::endl;
    }
    amrex::Finalize();
}
