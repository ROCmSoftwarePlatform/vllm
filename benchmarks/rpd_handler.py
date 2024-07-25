import subprocess

# Notes:
#  Cross-platform profiler: https://github.com/ROCmSoftwarePlatform/rocmProfileData/blob/master/FEATURES.md#rpd_tracer-startstop
#  CUDA-native profiler: https://nvidia.github.io/cuda-python/module/cudart.html#profiler-control

class HipTx:
    def __init__(self, platform=None, profname='rpd'):
        self.platform = (platform.upper() if platform else self.detect_platform())
        self.profname = profname  # can switch to 'nsight' for CUDA native method
        self.profbase = None

        if self.profname == 'rpd':
            # This is for cross-platform profiling
            from rpdTracerControl import rpdTracerControl
            rpdTracerControl.setFilename(name="trace.rpd", append=False)
            self.profbase = rpdTracerControl()
        elif self.profname == 'nsight':
            # This is for CUDA-only profiling
            assert self.platform == 'CUDA', "The platform must be 'CUDA' to use nsight profiling."
            from cuda import cudart
            self.profbase = cudart
        else:
            raise NotImplementedError("Profiler choice is not available.")

    def start_profiling(self):
        print("hiptx: start_profiling()")
        if self.profname == 'rpd':
            self.profbase.start()
        elif self.profname == 'nsight':
            self.profbase.cudaProfilerStart()

    def stop_profiling(self):
        print("hiptx: stop_profiling()")
        if self.profname == 'rpd':
            self.profbase.stop()
        elif self.profname == 'nsight':
            self.profbase.cudaProfilerStop()

    def start_range(self, msg):
        if self.profname == 'nsight':
            raise NotImplementedError("Range marking is not implemented yet.")
        else:
            raise NotImplementedError("Range marking is not implemented yet.")

    def stop_range(self, msg):
        if self.profname == 'nsight':
            raise NotImplementedError("Range marking is not implemented yet.")
        else:
            raise NotImplementedError("Range marking is not implemented yet.")

    def mark(self, msg):
        if self.profname == 'nsight':
            raise NotImplementedError("Marking is not implemented yet.")
        else:
            raise NotImplementedError("Marking is not implemented yet.")

    @staticmethod
    def detect_platform():
        try:
            subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return 'CUDA'
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        try:
            subprocess.run(['rocm-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return 'ROCM'
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        raise EnvironmentError("Neither ROCm nor CUDA platform is detected. Make sure the appropriate drivers are installed.")
