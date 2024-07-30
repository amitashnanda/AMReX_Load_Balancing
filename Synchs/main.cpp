
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <thread>
#include <future>

#ifdef AMREX_USE_GPU
#include <AMReX_Gpu.H>
#endif

struct gpu_sync {
    double tz;
    double sleep;
}

#ifdef AMREX_USE_CUDA
// Mocking GPU callback function work
void CUDART_CB do_step_gpu(void* p)
{
    auto p_sync = reinterpret_cast<gpu_sync*>(p);

    amrex::Print() << "Beginning callback work at " << amrex::second() - p_sync->tz << std::endl;
    amrex::Sleep(p_sync->sleep);
    amrex::Print() << "Ending callback work at " << amrex::second() - p_sync->tz << std::endl;
}
#endif



int main(int argc, char* argv[])
{
    // cudaDeviceScheduleAuto
    // cudaDeviceScheduleSpin
    // cudaDeviceScheduleYield
    // cudaDeviceScheduleBlockingSync

    // cudaSetDeviceFlags();

    amrex::Initialize(argc,argv);
    {
        amrex::Print() << "Hello world from AMReX version " << amrex::Version() << "\n";

        int warmups = 5;
        int tests = 1;
        double sleep = 0.1;
        double time_zero = amrex::second();

        amrex::Print() << "===========================================" << std::endl;
        amrex::Print() << "Starting at " << amrex::second() - time_zero << std::endl;

#ifdef AMREX_USE_CUDA
        for (int t=0; i<warmups+tests; ++i) {
            amrex::Print() << "*******************************************" << std::endl;
            amrex::Print() << "Loop number " << t << std::endl << std:endl;

            //  Single callback function
            //  ........................
            const gpu_sync p = new gpu_sync({time_zero, sleep});
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), do_step_gpu, (void*) p));

            amrex::Print() << "Callback launched at " << amrex::second() - time_zero << std::endl;
            amrex::Gpu::streamSynchronize();
            amrex::Print() << "Callback Sync done at " << amrex::second() - time_zero << std::endl;

            //  Single device kernel
            //  ....................

            amrex::Print() << std::endl;

            amrex::launch(1,
            [=] AMREX_GPU_DEVICE (long idx) {
                AMREX_DEVICE_PRINTF("Starting Kernel");
                __nanosleep(unsigned long(sleep*1e9));
                AMREX_DEVICE_PRINTF("Ending Kernel");
            });

            amrex::Print() << "Kernel launched at " << amrex::second() - time_zero << std::endl;
            amrex::Gpu::streamSynchronize();
            amrex::Print() << "Kernel Sync done at " << amrex::second() - time_zero << std::endl;

            //  MCMS (Multiple Callbacks, Multiple Streams) -- SyncAll
            //  ..................................................

            amrex::Print() << std::endl;

            amrex::Gpu::Device::setStreamIndex(0);
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), do_step_gpu, (void*) p));
            amrex::Gpu::Device::setStreamIndex(1);
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), do_step_gpu, (void*) p));
            amrex::Gpu::Device::setStreamIndex(2);
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), do_step_gpu, (void*) p));
            amrex::Gpu::Device::setStreamIndex(3);
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), do_step_gpu, (void*) p));

            amrex::Print() << "Multiple callbacks at " << amrex::second() - time_zero << std::endl;

            amrex::Gpu::streamSynchronizeAll();

            amrex::Print() << "DeviceSync completed at " << amrex::second() - time_zero << std::endl;

            amrex::Gpu::Device::setStreamIndex(0);
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), do_step_gpu, (void*) p));

            amrex::Print() << "Following callback at " << amrex::second() - time_zero << std::endl;
            amrex::Gpu::streamSynchronize(amrex::Gpu::gpuStream());
            amrex::Print() << "Complete set done at " << amrex::second() - time_zero << std::endl;

            //  MCMS -- nullStreamKernel
            //  ...........................................................

            amrex::Print() << std::endl;

            amrex::Gpu::Device::setStreamIndex(0);
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), do_step_gpu, (void*) p));
            amrex::Gpu::Device::setStreamIndex(1);
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), do_step_gpu, (void*) p));
            amrex::Gpu::Device::setStreamIndex(2);
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), do_step_gpu, (void*) p));
            amrex::Gpu::Device::setStreamIndex(3);
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), do_step_gpu, (void*) p));

            amrex::Print() << "Multiple callbacks at " << amrex::second() - time_zero << std::endl;

            amrex::setStreamIndex(-1);
            amrex::launch(1,
            [=] AMREX_GPU_DEVICE (long idx) {
                AMREX_DEVICE_PRINTF("Starting Sync Kernel");
                __nanosleep(unsigned long(sleep*1e9));
                AMREX_DEVICE_PRINTF("Ending Sync Kernel");
            });

            amrex::Print() << "Null stream at " << amrex::second() - time_zero << std::endl;

            amrex::Gpu::Device::setStreamIndex(0);
            AMREX_CUDA_SAFE_CALL(cudaLaunchHostFunc(amrex::Gpu::gpuStream(), do_step_gpu, (void*) p));

            amrex::Print() << "Following callback at " << amrex::second() - time_zero << std::endl;
            amrex::Gpu::streamSynchronize(amrex::Gpu::gpuStream());
            amrex::Print() << "Complete set done at " << amrex::second() - time_zero << std::endl;

            amrex::Print() << "===========================================" << std::endl;

            delete p;
        }
#endif
    }
    amrex::Finalize();
}

