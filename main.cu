
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <math.h>

#include <chrono>

#define DEBUG 1

#define NULL_ADDRESS_MARGIN 4
#define NULL_ADDRESS_READ (0x10000 + 0)
#define NULL_ADDRESS_WRITE (0x10000 + 2)
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

__device__
void operationTransition(uint8_t opcode, SystemState* state, ComputationState* computation_state) {
    instructions[opcode].transition(state, computation_state);
}

/* */

__global__
void add(int num_states, uint32_t num_instructions, SystemState *states)
{
    for (int i = 0; i < num_instructions; i++) {
        states[threadIdx.x].next();
    }
}

#include "utilities.h"

/* */

int tests(void)
{
    // int num_states = 256;
    int num_states = 15;
    SystemState *states;

    uint64_t num_instructions = 0;
    std::cin >> num_instructions;

    std::cout << "NUM_INSTRUCTIONS [ " << num_instructions << ", " << sizeof(SystemState) << " ]\n\n";

    cudaMallocManaged(&states, num_states*sizeof(SystemState));

    /* */

    std::vector<char> program_data = romFileRead("roms/nestest.nes").first;
    std::vector<std::vector<std::string>> log_lines = logRead("data/nestest.log");

    for (int i = 0; i < num_states; i++) {
        states[i] = SystemState(program_data, 0xC000 + i - 7, 0xC000);
    }

    /* */

    auto start = std::chrono::high_resolution_clock::now();

    add<<<1, num_states>>>(num_states, num_instructions, states);
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

        std::cout << traceLineFormat(states[i].traceLineData[states[i].traceIndex-1]) << "\n";
    }

    std::cout << "\n";

    /* */

    int mismatch_count = 0;
    for (int i = 0; i < states[7].traceIndex; i++) {
        std::string reference = logLineFormat(log_lines[i]);
        std::string actual = traceLineFormat(states[7].traceLineData[i]);

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
    // int num_states = 256;
    int num_states = 15;
    SystemState *states;

    uint64_t num_instructions = 0;
    std::cin >> num_instructions;

    std::cout << "NUM_INSTRUCTIONS [ " << num_instructions << ", " << sizeof(SystemState) << " ]\n\n";

    cudaMallocManaged(&states, num_states*sizeof(SystemState));

    /* */

    auto program_data = romFileRead("roms/Super Mario Bros..nes").first;
    auto character_data = romFileRead("roms/Super Mario Bros..nes").second;

    std::vector<std::vector<std::string>> log_lines = logRead("data/nestest.log");

    for (int i = 0; i < num_states; i++) {
        if (i == 7) {
            states[i] = SystemState(program_data);
        } else {
            // states[i] = SystemState(program_data, 0xC000 + i - 7, 0xC000);
            states[i] = SystemState(program_data);
        }
    }

    /* */

    auto start = std::chrono::high_resolution_clock::now();

    add<<<1, num_states>>>(num_states, num_instructions, states);
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

        std::cout << traceLineFormat(states[i].traceLineData[states[i].traceIndex-1]) << "\n";
    }

    std::cout << "\n";

    /* */

    for (int i = 0; i < states[7].traceIndex; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << std::uppercase << i << " ";
        std::cout << "│ " << traceLineFormat(states[7].traceLineData[i], true) << "\n";
    }

    std::cout << std::endl;

    int8u_t* ppu_data = &states[7].memory.array[PPU_OFFSET + 0x2000];

    printf("0x2000\n");

    for (int i = 0; i < 1024; i++) {
        printf(" 0x%02X", ppu_data[i]);

        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }

    printf("\n");
    printf("\n");

    int8u_t* ppu_data2 = &states[7].memory.array[PPU_OAM_OFFSET];

    printf("0x%04X\n", PPU_OAM_OFFSET);

    for (int i = 0; i < 256; i++) {
        printf(" 0x%02X", ppu_data2[i]);

        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }

    printf("\n");

    #endif

    /* */

    cudaFree(states);

    return 0;
}

int main(void)
{
    // return tests();
    return software();
}