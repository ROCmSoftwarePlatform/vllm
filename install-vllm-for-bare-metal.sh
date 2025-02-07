#!/usr/bin/env bash

# READ BEFORE USE:
# This script helps to install vllm on bare-metal. The time it takes is in term of hours.

# The script default installs pytorch and vllm, and assumes rocm and its environment has been installed.
# The vllm also has other dependencies, if you find something is missing you can uncomment certain script block to install.
# Make sure the vllm whl to install is in the same directory as this script. 
# The script may also pull some repository in current directory, you can delete them after installation.

# To test on vllm installation: run some examples in vllm/examples.





# # Install some basic utilities
# apt-get update -q -y && apt-get install -q -y python3 python3-pip
# apt-get update -q -y && apt-get install -q -y sqlite3 libsqlite3-dev libfmt-dev libmsgpack-dev libsuitesparse-dev
# # Remove sccache    
# python3 -m pip install --upgrade pip
# apt-get purge -y sccache; python3 -m pip uninstall -y sccache; rm -f "$(which sccache)"


# # local build environment
# export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
# export LLVM_SYMBOLIZER_PATH="/opt/rocm/llvm/bin/llvm-symbolizer"
# export PATH="$PATH:/opt/rocm/bin:/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/bin:"
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/rocm/lib/:/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/lib:"
# export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/include:/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/:/opt/rocm/include/:"


# python3 -m pip install --upgrade pip && rm -rf /var/lib/apt/lists/*


# # optional numpy update
# case "$(which python3)" in \
#     *"/opt/conda/envs/py_3.9"*) \
#         rm -rf /opt/conda/envs/py_3.9/lib/python3.9/site-packages/numpy-1.20.3.dist-info/;; \
#     *) ;; esac


# # optional flash-attention installation
# FA_BRANCH="3cea2fb"
# pip uninstall -y flash-attn
# git clone https://github.com/ROCm/flash-attention.git
# cd flash-attention
# git checkout $FA_BRANCH
# git submodule update --init
# GPU_ARCHS=${PYTORCH_ROCM_ARCH} python3 setup.py bdist_wheel --dist-dir=dist
# pip install dist/*.whl
# cd ..


# # optional triton installation
# pip uninstall -y triton \
# && python3 -m pip install ninja cmake wheel pybind11 && git clone https://github.com/triton-lang/triton.git \
# && cd triton \
# && git checkout e192dba \
# && cd python \
# && python3 setup.py bdist_wheel --dist-dir=dist \
# && pip install dist/*.whl \
# && cd .. && cd ..


# pytorch installation
pip uninstall -y torch torchvision \
&& git clone https://github.com/ROCm/pytorch.git pytorch \
&& cd pytorch && git checkout cedc116 && git submodule update --init --recursive \
&& python3 tools/amd_build/build_amd.py \
&& CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)') python3 setup.py bdist_wheel --dist-dir=dist \
&& pip install dist/*.whl \
&& cd .. \
&& git clone https://github.com/pytorch/vision.git vision \
&& cd vision && git checkout v0.19.1 \
&& python3 setup.py bdist_wheel --dist-dir=dist \
&& pip install dist/*.whl \
&& cd ..


# # optional huggingface installation
# python3 -m pip install --upgrade huggingface-hub[cli]


# # optional profile installation
# git clone -b nvtx_enabled https://github.com/ROCm/rocmProfileData.git \
# && cd rocmProfileData/rpd_tracer \
# && pip install -r requirements.txt && cd ../ \
# && make && make install \
# && cd hipMarker && python setup.py install \
# && cd ../..


# Install vLLM (and gradlib)
pip install -U -r requirements-rocm.txt \
&& pip uninstall -y vllm gradlib \
&& pip install *.whl


# Install amd_smi
# cd /opt/rocm/share/amd_smi \
# && pip wheel . --wheel-dir=dist \
# && pip install dist/*.whl \
