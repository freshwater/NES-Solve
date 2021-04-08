
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <chrono>

#define DO_SOFTWARE true
#define OBSERVED_INSTANCE 7
#define FRAMEDATA_SIZE 256*240
// #define DEBUG 1

typedef uint16_t int16u_t;
typedef uint8_t int8u_t;

struct SystemState;
struct ComputationState;

#include "states.h"
#include "regions.h"
#include "_instructions.h"
#include "utilities.h"

__device__
void operationTransition(uint8_t opcode, SystemState* system, ComputationState* state, Memory& memory) {
    instructions[opcode].transition(system, state, memory);
}

namespace NESSolveModule {
    __global__
    __launch_bounds__(32) // , minBlocksPerMultiprocessor)
    void add(SystemState* systems, uint8_t* actions, int num_actions)
    {
        int instance_index = blockIdx.x*(blockDim.x) + threadIdx.x;

        ComputationState state;
        state.program_counter = systems[instance_index].program_counter_initial;
        state.stack_offset = systems[instance_index].stack_offset_initial;
        state.num_actions = num_actions;

        Memory memory;
        for (int i = 0; i < sizeof(Memory); i++) {
            memory.array[i] = systems[instance_index].global_memory.array[i];
        }

        while (state.frame_count < num_actions) {
            int frame_index = state.frame_count*(gridDim.x * blockDim.x) +
                              blockIdx.x*(blockDim.x) +
                              threadIdx.x;

            state.control_port1 = actions[frame_index];
            systems[instance_index].next(&state, memory);
        }
    }

    void run(const unsigned char* file_location, int file_location_size,
             char* _actions, int num_instances, int num_actions, int num_blocks,
             char* frames_red_out, char* frames_green_out, char* frames_blue_out)
    {
        SystemState *systems;
        uint8_t* actions;
        uint8_t* frames_red;
        uint8_t* frames_green;
        uint8_t* frames_blue;

        cudaMallocManaged(&systems, num_instances*sizeof(SystemState));
        cudaMallocManaged(&actions, num_instances*sizeof(SystemState));
        cudaMallocManaged(&frames_red, num_instances*FRAMEDATA_SIZE);
        cudaMallocManaged(&frames_green, num_instances*FRAMEDATA_SIZE);
        cudaMallocManaged(&frames_blue, num_instances*FRAMEDATA_SIZE);

        std::string file((char *)file_location, file_location_size);
        std::vector<char> program_data1 = romFileRead(file).first;
        std::vector<char> character_data = romFileRead(file).second;

        printf("\nRegionComposition(%d)\n", sizeof(RegionComposition));
        printf("instructions(%d)\n\n", sizeof(instructions));

        auto mark1 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_instances; i++) {
            systems[i] = SystemState(program_data1, character_data);
            systems[i].frames_red = frames_red;
            systems[i].frames_green = frames_green;
            systems[i].frames_blue = frames_blue;
        }
        memcpy(actions, _actions, num_instances*num_actions);

        auto mark2 = std::chrono::high_resolution_clock::now();

        int num_threads_per_block = num_instances / num_blocks;

        printf("<<<%d, %d>>>(%d, _, %d)\n", num_blocks, num_threads_per_block, num_instances, num_actions);
        add<<<num_blocks, num_threads_per_block>>>(systems, actions, num_actions);
        cudaDeviceSynchronize();

        auto mark3 = std::chrono::high_resolution_clock::now();
        std::cout << "\nstates> " << std::chrono::duration_cast<std::chrono::microseconds>(mark2 - mark1).count();
        std::cout << "\n total> " << std::chrono::duration_cast<std::chrono::microseconds>(mark3 - mark1).count();
        std::cout << "\n   add> " << std::chrono::duration_cast<std::chrono::microseconds>(mark3 - mark2).count() << std::endl;

        memcpy(frames_red_out, frames_red, num_instances*FRAMEDATA_SIZE);
        memcpy(frames_green_out, frames_green, num_instances*FRAMEDATA_SIZE);
        memcpy(frames_blue_out, frames_blue, num_instances*FRAMEDATA_SIZE);

        cudaFree(&systems);
        cudaFree(&frames_red);
        cudaFree(&frames_green);
        cudaFree(&frames_blue);
    }
}

int main() {

}
