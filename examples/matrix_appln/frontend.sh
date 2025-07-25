export EXTRA_NVCCFLAGS="--cudart=shared"
export GVIRTUS_LOGLEVEL=20000
export GVIRTUS_HOME=/home/darshan/GVirtuS
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
nvcc simple_matrix.cu -o example -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ -lcuda --cudart=shared 
./example
