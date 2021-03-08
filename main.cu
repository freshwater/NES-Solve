
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <math.h>

#include <chrono>

#define DEBUG 1
#define MAXFRAMES 60*10

#define NULL_ADDRESS_MARGIN 4
#define STACK_ZERO 0x0100

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

/* */

__global__
void add(uint32_t num_instructions, SystemState *states)
{
    ComputationState computation_state;
    computation_state.program_counter = states[threadIdx.x].program_counter_initial;
    computation_state.stack_offset = states[threadIdx.x].stack_offset_initial;

    for (int i = 0; i < num_instructions; i++) {
        /*if (160 < computation_state.frame_count && computation_state.frame_count < 200) {
            states[threadIdx.x].memory.control_port1[0] = 0x10;
            // states[threadIdx.x].memory.control_port1[0] = 0x00;
        } else {
            states[threadIdx.x].memory.control_port1[0] = 0x00;
        }*/

        states[threadIdx.x].next(&computation_state);

        if (computation_state.frame_count == MAXFRAMES) {
            break;
        }
    }

    if (threadIdx.x == 7) {
        printf("FrameCount(%d)\n", computation_state.frame_count);
    }
}

/* */

int tests(void)
{
    int num_states = 15;
    int num_trace_lines = 100000;
    SystemState *states;
    Trace *trace_lines;

    uint64_t num_instructions = 0;
    std::cin >> num_instructions;

    std::cout << "NUM_INSTRUCTIONS [ " << num_instructions << ", " << sizeof(SystemState) << " ]\n";

    cudaMallocManaged(&states, num_states*sizeof(SystemState));
    cudaMallocManaged(&trace_lines, num_trace_lines*sizeof(Trace));

    /* */

    std::vector<char> program_data = romFileRead("roms/nestest.nes").first;
    std::vector<std::vector<std::string>> log_lines = logRead("data/nestest.log");

    for (int i = 0; i < num_states; i++) {
        states[i] = SystemState(program_data, 0xC000 + i - 7, 0xC000);
        states[i].trace_lines = trace_lines;
    }

    /* */

    auto start = std::chrono::high_resolution_clock::now();

    add<<<1, num_states>>>(num_instructions, states);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "\n> " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << "\n\n";

    /* */

    #ifdef DEBUG

    std::cout << "\n";
    for (int i = 0; i < num_states; i++) {
        if (i == 7 || i == 8) {
            std::cout << "--------------------------" << "\n";
        }

        std::cout << traceLineFormat(states[i].traceLineLast) << "\n";
    }

    std::cout << "\n";

    /* */

    int mismatch_count = 0;
    for (int i = 0; i < states[7].trace_lines_index; i++) {
        std::string reference = logLineFormat(log_lines[i]);
        std::string actual = traceLineFormat(trace_lines[i], false);

        if (reference == actual) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << std::uppercase << i << " ";
            std::cout << "│ " << actual << "\n";
        } else {
            std::cout << "\n" << std::hex << std::setw(2) << std::setfill('0') << std::uppercase << i << "";
            std::cout << " · " << reference << "\n";
            std::cout << "     " << lineCompare(reference, actual) << "\n";

            mismatch_count++;
        }
    }

    std::cout << "\n" << mismatch_count << "\n" << std::endl;

    #endif

    /* */

    cudaFree(states);

    return 0;
}

int software(void)
{
    int num_states = 15;
    int num_trace_lines = 0;
    SystemState *states;
    Trace *trace_lines;

    uint64_t num_instructions = 100000000;
    // std::cin >> num_instructions;

    std::cout << "NUM_INSTRUCTIONS [ " << num_instructions << ", " << sizeof(SystemState) << " ]\n\n";
    std::cout << "SIZEOF(RegionComposition)=" << sizeof(RegionComposition) << "\n";
    std::cout << "SIZEOF(_instructions)=" << sizeof(instructions) << "\n\n";

    std::cout << std::endl;

    cudaMallocManaged(&states, num_states*sizeof(SystemState));
    cudaMallocManaged(&trace_lines, num_trace_lines*sizeof(Trace));

    /* */

    std::string game = "Donkey Kong.nes";
    // std::string game = "nestest.nes";
    // std::string game = "Bubble Bobble (U).nes";
    // std::string game = "Super Mario Bros..nes";
    // std::string game = "Spelunker (USA).nes";
    auto program_data = romFileRead("roms/" + game).first;
    auto character_data = romFileRead("roms/" + game).second;

    std::vector<std::vector<std::string>> log_lines = logRead("data/nestest.log");

    for (int i = 0; i < num_states; i++) {
        states[i] = SystemState(program_data, character_data);
        states[i].trace_lines = trace_lines;
    }

    /* */

    auto start = std::chrono::high_resolution_clock::now();

    // add<<<1, num_states>>>(num_instructions, states);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "\n> " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << "\n\n";

    std::vector<char> data1(std::begin(states[7].memory.ppu_memory), std::end(states[7].memory.ppu_memory));
    std::vector<char> data2(std::begin(states[7].memory.ppu_OAM_memory), std::end(states[7].memory.ppu_OAM_memory));

    imageWrite(data1, data2);

    /* */

    #ifdef DEBUG

    std::cout << "\n";
    for (int i = 0; i < num_states; i++) {
        if (i == 7 || i == 8) {
            std::cout << "--------------------------" << "\n";
        }

        std::cout << traceLineFormat(states[i].traceLineLast) << "\n";
    }

    std::cout << "\n";

    /* */

    for (int i = 0; i < states[7].trace_lines_index; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << std::uppercase << i << " ";
        std::cout << "│ " << traceLineFormat(trace_lines[i], true) << "\n";
    }

    std::cout << std::endl;

    #endif

    /* */

    cudaFree(states);

    return 0;
}

int main(void)
{
    return tests();
    // return software();
}