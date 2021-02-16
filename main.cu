
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <math.h>

#define NULL_ADDRESS_MARGIN 4
#define NULL_ADDRESS_READ (0x10000 + 0)
#define NULL_ADDRESS_WRITE (0x10000 + 2)

typedef const uint32_t flag_t;
typedef const uint32_t int_t;
typedef uint32_t int16u_t;
typedef uint32_t int8u_t;
typedef uint8_t int8u_memory_t;

struct SystemState;
struct ComputationState;

__device__
void operationTransition(uint8_t, SystemState*, ComputationState*);

#include "states.h"
#include "regions.h"
#include "_instructions.h"

__device__
void operationTransition(uint8_t opcode, SystemState* state, ComputationState* computation_state) {
    instructions[opcode].transition(state, computation_state);
}

/* */

__global__
void add(int num_states, SystemState *states)
{
    states[threadIdx.x].next();
    states[threadIdx.x].next();
    states[threadIdx.x].next();
    states[threadIdx.x].next();
    states[threadIdx.x].next();
}

#include "utilities.h"

/* */

int main(void)
{
    int num_states = 15;
    SystemState *states;

    cudaMallocManaged(&states, num_states*sizeof(SystemState));

    /* */

    std::vector<char> file_data = fileRead("data/nestest.program");
    std::vector<std::vector<std::string>> log_lines = logRead("data/nestest.log");

    for (int i = 0; i < num_states; i++) {
        states[i] = SystemState(file_data, 0xC000 + i - 7, 0xC000);
    }

    /* */

    add<<<1, num_states>>>(num_states, states);
    cudaDeviceSynchronize();

    /* */

    std::cout << "\n";
    for (int i = 0; i < num_states; i++) {
        if (i == 7 || i == 8) {
            std::cout << "--------------------------" << "\n";
        }

        std::cout << traceLineFormat(states[i].traceLineData[states[i].traceIndex-1]) << "\n";
    }

    std::cout << "\n";

    /* */

    for (int i = 0; i < states[7].traceIndex; i++) {
        std::cout << "R " << logLineFormat(log_lines[i]) << "\n";
        std::cout << "A " << traceLineFormat(states[7].traceLineData[i]) << "\n";
        std::cout << "> " << lineCompare(logLineFormat(log_lines[i]), traceLineFormat(states[7].traceLineData[i])) << "\n";
        std::cout <<"\n";
    }

    /* */

    cudaFree(states);

    return 0;
}