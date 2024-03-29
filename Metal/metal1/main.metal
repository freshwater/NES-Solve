//
//  main.metal
//

#include "states.h"

kernel void run(device SystemState* systemStates [[buffer(0)]], device Trace* traceLines [[buffer(1)]]) {
    ComputationState state;
    state.program_counter = systemStates[0].program_counter_initial;
    state.stack_offset = systemStates[0].stack_offset_initial;

    for (int i = 0; i < 3*0x1800; i++) {
        SystemState__next(systemStates[0], &state, systemStates[0].global_memory, traceLines);
    }
}
