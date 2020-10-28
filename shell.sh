
set -e

PWD_=$PWD
cd $(dirname $0)

# docker exec --interactive --tty $(docker ps | grep question-mark-shell | awk '{print $1}') bash

# TAG=tag$RANDOM
TAG=question-mark-shell
docker build . --tag $TAG && docker run --rm --interactive --tty --mount type=bind,src="$PWD_",dst=/workfolder $TAG bash
