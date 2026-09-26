# Headless stella_vslam build (no viewer) used by slam3d as the equirectangular SLAM backend.
# Upstream is built unmodified at pinned commits; slam3d's driver (cpp/stella_driver) is compiled
# against the installed library at run time via a mounted volume.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive DEBCONF_NOWARNINGS=yes
RUN apt-get update -y -qq && apt-get install -y -qq --no-install-recommends \
      build-essential pkg-config cmake git ca-certificates wget \
      libeigen3-dev libopencv-dev libyaml-cpp-dev libsuitesparse-dev \
      sqlite3 libsqlite3-dev binutils-dev \
    && rm -rf /var/lib/apt/lists/*

ARG NUM_THREADS=16
ARG G2O_COMMIT=20230223_git
WORKDIR /tmp
RUN git clone -q https://github.com/RainerKuemmerle/g2o.git && cd g2o && git checkout -q ${G2O_COMMIT} && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_UNITTESTS=OFF \
      -DG2O_USE_CHOLMOD=OFF -DG2O_USE_CSPARSE=ON -DG2O_USE_OPENGL=OFF -DG2O_USE_OPENMP=OFF \
      -DG2O_BUILD_APPS=OFF -DG2O_BUILD_EXAMPLES=OFF -DG2O_BUILD_LINKED_APPS=OFF .. && \
    make -j${NUM_THREADS} && make install && cd /tmp && rm -rf g2o

# g2o's installed g2oConfig.cmake calls find_dependency(OpenGL) even when G2O_USE_OPENGL=OFF
RUN apt-get update -y -qq && apt-get install -y -qq --no-install-recommends libgl1-mesa-dev libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

ARG STELLA_COMMIT=e445b5452535e781fdf6777ec33a06f5f8f5e416
RUN git clone -q https://github.com/stella-cv/stella_vslam.git /stella_vslam && cd /stella_vslam && \
    git checkout -q ${STELLA_COMMIT} && git submodule update --init --recursive -q && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. && \
    make -j${NUM_THREADS} && make install && ldconfig

ENV LD_LIBRARY_PATH=/usr/local/lib
WORKDIR /work
