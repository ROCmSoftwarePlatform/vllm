#!/bin/bash
# install dependency
apt update && apt install -y sqlite3 libsqlite3-dev libfmt-dev

# install rpd module
git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData && \
    cd rocmProfileData && \
    make && sudo make install && \
    cd rocpd_python && python setup.py install && cd .. && \
    cd rpd_tracer && python setup.py install && cd ..
