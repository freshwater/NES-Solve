
FROM ubuntu:latest

RUN apt-get update --fix-missing

RUN apt-get install --yes wget
RUN apt-get install --yes lsb-release \
                          software-properties-common

RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes erlang

COPY llvm.sh .
RUN chmod +x llvm.sh
RUN ./llvm.sh

WORKDIR workfolder

ENV PATH="/bin:${PATH}"
