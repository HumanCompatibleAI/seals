#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.
set -x # echo on

__usage="launch_docker-dev.sh - Launching humancompatibleai/seals:python-req
Usage: launch_docker-dev.sh [options] 
options:
  -p, --pull                pull the image to DockerHub
  -h, --help                show this help message and exit
Note: You can specify LOCAL_MNT environment variables to mount local repository
    and output directory respectively.
"

PULL=0

while test $# -gt 0; do
  case "$1" in
  -p | --pull)
    PULL=1 # Pull the image from Docker Hub
    shift
    ;;
  -h | --help)
    echo "${__usage}"
    exit 0
    ;;
  *)
    echo "Unrecognized flag $1" >&2
    exit 1
    ;;
  esac
done

##################
## Docker Image ##
##################

DOCKER_IMAGE="humancompatibleai/seals:python-req"
if [[ ${LOCAL_MNT} == "" ]]; then
  LOCAL_MNT="${HOME}"
fi

###########
## Flags ##
###########

# if port is changed here, it should also be changed in scripts/launch_jupyter.sh
FLAGS+="--gpus all "  # Use all GPUs
FLAGS+="-p 9998:9998 "  # ports
FLAGS+="-v ${LOCAL_MNT}/seals:/seals "  # mounting local seals repo

##############
## Commands ##
##############

CMD+="pip3 install -e .[dev,mujoco] "

# Using jupyter lab for easy development
if [[ $1 == "jupyter" ]]; then
  CMD+="&& scripts/launch_jupyter.sh "
fi

####################################
## (Pull image and) Run Container ##
####################################

# Pull image from DockerHub if prompted
if [[ $PULL == 1 ]]; then
  echo "Pulling ${DOCKER_IMAGE} from DockerHub"
  docker pull ${DOCKER_IMAGE}
fi

docker run -it --rm --init --name seals \
       ${FLAGS} \
       ${DOCKER_IMAGE} \
       /bin/bash -c "${CMD} && exec bash"