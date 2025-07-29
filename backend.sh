#!/bin/bash

# Exit on error
set -e

# Set GVirtuS home directory
export GVIRTUS_HOME=/home/darshan/GVirtuS

# Run gvirtus-backend with updated LD_LIBRARY_PATH
LD_LIBRARY_PATH="${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}" \
${GVIRTUS_HOME}/bin/gvirtus-backend ${GVIRTUS_HOME}/etc/properties.json

