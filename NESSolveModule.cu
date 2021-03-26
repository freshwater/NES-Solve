
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <chrono>

#define DO_SOFTWARE true
#define MAXFRAMES (60*2)
#define OBSERVED_INSTANCE 7
#define FRAMEDATA_SIZE 256*240
// #define DEBUG 1

typedef const uint8_t flag_t;
typedef const uint16_t flag16_t;
typedef const uint8_t int_t;
typedef const int8_t int_signed_t;
typedef uint16_t int16u_t;
typedef uint8_t int8u_t;
typedef uint8_t bit_t;

struct SystemState;
struct ComputationState;

__device__
void operationTransition(uint8_t, SystemState*, ComputationState*);

#include "states.h"
#include "regions.h"
#include "_instructions.h"
#include "utilities.h"

__device__
void operationTransition(uint8_t opcode, SystemState* state, ComputationState* computation_state) {
    instructions[opcode].transition(state, computation_state);
}

namespace NESSolveModule {
    __global__
    void add(SystemState* states)
    {
        ComputationState computation_state;
        computation_state.program_counter = states[threadIdx.x].program_counter_initial;
        computation_state.stack_offset = states[threadIdx.x].stack_offset_initial;

        while (computation_state.frame_count < MAXFRAMES) {
            states[threadIdx.x].next(&computation_state);
        }

        if (threadIdx.x == OBSERVED_INSTANCE) {
            printf("FrameCount(%d)\n", computation_state.frame_count);
            printf("NMI(%d)\n", computation_state.nmi_count);
        }
    }

    void run(const unsigned char* file_location, int file_location_size, char* frames_red_out)
    {
        int num_states = 15;
        int num_trace_lines = 0;
        SystemState *states;
        uint8_t* frames_red;
        uint8_t* frames_green;
        uint8_t* frames_blue;

        cudaMallocManaged(&states, num_states*sizeof(SystemState));
        cudaMallocManaged(&frames_red, FRAMEDATA_SIZE);
        cudaMallocManaged(&frames_green, FRAMEDATA_SIZE);
        cudaMallocManaged(&frames_blue, FRAMEDATA_SIZE);

        std::string file((char *)file_location, file_location_size);
        std::vector<char> program_data = romFileRead(file).first;
        std::vector<char> character_data = romFileRead(file).second;

        auto mark1 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_states; i++) {
            states[i] = SystemState(program_data, character_data);
        }

        auto mark2 = std::chrono::high_resolution_clock::now();

        states[OBSERVED_INSTANCE].frames_red = frames_red;
        states[OBSERVED_INSTANCE].frames_green = frames_green;
        states[OBSERVED_INSTANCE].frames_blue = frames_blue;

        add<<<1, num_states>>>(states);
        cudaDeviceSynchronize();

        auto mark3 = std::chrono::high_resolution_clock::now();
        std::cout << "\nstates> " << std::chrono::duration_cast<std::chrono::microseconds>(mark2 - mark1).count();
        std::cout << "\n   add> " << std::chrono::duration_cast<std::chrono::microseconds>(mark3 - mark2).count();
        std::cout << "\n total> " << std::chrono::duration_cast<std::chrono::microseconds>(mark3 - mark1).count() << std::endl;

        memcpy(frames_red_out, frames_red, FRAMEDATA_SIZE);

        cudaFree(&states);
        cudaFree(&frames_red);
        cudaFree(&frames_green);
        cudaFree(&frames_blue);
    }
}
