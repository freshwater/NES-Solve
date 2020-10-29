
FROM ubuntu:latest

RUN apt-get update --fix-missing

RUN apt-get install --yes wget
RUN apt-get install --yes lsb-release \
                          software-properties-common

COPY llvm.sh .
RUN chmod +x llvm.sh
RUN ./llvm.sh

RUN wget -O - https://packages.erlang-solutions.com/ubuntu/erlang_solutions.asc | apt-key add -
RUN echo "deb https://packages.erlang-solutions.com/ubuntu focal contrib" | tee /etc/apt/sources.list.d/rabbitmq.list
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes erlang

WORKDIR workfolder

ENV PATH="/bin:${PATH}"
