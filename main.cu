
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <math.h>

#include <chrono>

#define DO_SOFTWARE true
#define MAXFRAMES (60*11 + 30)
#define OBSERVED_INSTANCE 7
#define FRAMEDATA_SIZE 256*240
// #define DEBUG 1

#define SOFTWARE "Donkey Kong.nes"
// #define SOFTWARE "nestest.nes"
// #define SOFTWARE "01-basics.nes"
// #define SOFTWARE "digdug.nes"
// #define SOFTWARE "Spy vs Spy (USA).nes"
// #define SOFTWARE "Bubble Bobble (U).nes"
// #define SOFTWARE "Super Mario Bros..nes"
// #define SOFTWARE "Mario Bros (JU).nes"
// #define SOFTWARE "Spelunker (USA).nes"

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
        if (30*4 < computation_state.frame_count && computation_state.frame_count < 30*5 + 10) {
            states[threadIdx.x].memory.control_port1[0] = 0x10;
        } else if (30*21 <= computation_state.frame_count && computation_state.frame_count < 30*22) {
            states[threadIdx.x].memory.control_port1[0] = 0x01;
        } else if (30*22 <= computation_state.frame_count && computation_state.frame_count < 30*23) {
            states[threadIdx.x].memory.control_port1[0] = 0x81;
        } else {
            states[threadIdx.x].memory.control_port1[0] = 0x00;
        }

        states[threadIdx.x].next(&computation_state);

        if (computation_state.frame_count == MAXFRAMES) {
            break;
        }
    }

    if (threadIdx.x == OBSERVED_INSTANCE) {
        printf("FrameCount(%d)\n", computation_state.frame_count);
        printf("NMI(%d)\n", computation_state.nmi_count);
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
        states[i] = SystemState(program_data, 0xC000 + i - OBSERVED_INSTANCE, 0xC000);
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
        if (i == OBSERVED_INSTANCE || i == OBSERVED_INSTANCE + 1) {
            std::cout << "--------------------------" << "\n";
        }

        std::cout << traceLineFormat(states[i].traceLineLast) << "\n";
    }

    std::cout << "\n";

    /* */

    int mismatch_count = 0;
    for (int i = 0; i < states[OBSERVED_INSTANCE].trace_lines_index; i++) {
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
    cudaFree(trace_lines);

    return 0;
}

int software(void)
{
    int num_states = 15;
    int num_trace_lines = 0;
    SystemState *states;
    Trace *trace_lines;
    uint8_t* frames_red;
    uint8_t* frames_green;
    uint8_t* frames_blue;

    uint64_t num_instructions = 100000000;
    // std::cin >> num_instructions;

    std::cout << "\n[]\n\n" << std::endl;

    cudaMallocManaged(&states, num_states*sizeof(SystemState));
    cudaMallocManaged(&trace_lines, num_trace_lines*sizeof(Trace));
    cudaMallocManaged(&frames_red, FRAMEDATA_SIZE);
    cudaMallocManaged(&frames_green, FRAMEDATA_SIZE);
    cudaMallocManaged(&frames_blue, FRAMEDATA_SIZE);

    /* */

    std::string software = SOFTWARE;

    auto program_data = romFileRead("roms/" + software).first;
    auto character_data = romFileRead("roms/" + software).second;

    std::vector<std::vector<std::string>> log_lines = logRead("data/nestest.log");

    for (int i = 0; i < num_states; i++) {
        states[i] = SystemState(program_data, character_data);
    }

    states[OBSERVED_INSTANCE].trace_lines = trace_lines;
    states[OBSERVED_INSTANCE].frames_red = frames_red;
    states[OBSERVED_INSTANCE].frames_green = frames_green;
    states[OBSERVED_INSTANCE].frames_blue = frames_blue;

    memset(frames_red, 111, FRAMEDATA_SIZE);

    /* */

    auto start = std::chrono::high_resolution_clock::now();

    add<<<1, num_states>>>(num_instructions, states);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "\n> " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << "\n\n";

    std::vector<char> data1(std::begin(states[OBSERVED_INSTANCE].memory.ppu_memory), std::end(states[OBSERVED_INSTANCE].memory.ppu_memory));
    std::vector<char> data2(std::begin(states[OBSERVED_INSTANCE].memory.ppu_OAM_memory), std::end(states[OBSERVED_INSTANCE].memory.ppu_OAM_memory));

    std::vector<char> data3(FRAMEDATA_SIZE);
    std::vector<char> data4(FRAMEDATA_SIZE);
    std::vector<char> data5(FRAMEDATA_SIZE);
    memcpy(data3.data(), frames_red, FRAMEDATA_SIZE);
    memcpy(data4.data(), frames_green, FRAMEDATA_SIZE);
    memcpy(data5.data(), frames_blue, FRAMEDATA_SIZE);

    imageWrite(data1, data2, data3, data4, data5);

    /* */

    #ifdef DEBUG

    std::cout << "\n";

    for (int i = 0; i < states[OBSERVED_INSTANCE].trace_lines_index; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << std::uppercase << i << " ";
        std::cout << "│ " << traceLineFormat(trace_lines[i], true) << "\n";
    }

    std::cout << std::endl;

    #endif

    /* */

    cudaFree(states);
    cudaFree(trace_lines);
    cudaFree(frames_red);
    cudaFree(frames_green);
    cudaFree(frames_blue);

    return 0;
}

int main(void)
{
    if (DO_SOFTWARE) {
        return software();
    } else {
        return tests();
    }
}