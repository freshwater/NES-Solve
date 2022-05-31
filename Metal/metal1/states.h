//
//  states.h
//

#ifdef __METAL_VERSION__
    #define __METAL__
#else
    #define __SWIFT__
#endif

#ifdef __SWIFT__
    #import <Foundation/Foundation.h>
#endif

#define int16u_t uint16_t
#define int8u_t uint8_t

struct Trace {
    int16u_t program_counter;
    int8u_t opcode;
    int8u_t byte1;
    int8u_t byte2;

    int8u_t A;
    int8u_t X;
    int8u_t Y;

    int8u_t status_register;
    int8u_t stack_offset;

    int16_t horizontal_scan;

    /*
    int16_t vertical_scan;
    int16_t horizontal_scan;
    int16u_t cycle;*/
};

#ifdef __METAL__
struct ComputationState {
    int16u_t program_counter;
    int8u_t stack_offset;

    int8u_t A = 0;
    int8u_t X = 0;
    int8u_t Y = 0;

    int8u_t N = 0;
    int8u_t O = 0;
    int8u_t U = 1;
    int8u_t B = 0;
    int8u_t D = 0;
    int8u_t I = 1;
    int8u_t Z = 0;
    int8u_t C = 0;

    int8u_t opcode;
    int8u_t data1;
    int8u_t data2;

    int8u_t value1;
    int8u_t value2;
    int8u_t value3;

    int16u_t address;

    /*
    int8u_t store;
    int8u_t ppu_status;
    */

    int horizontal_scan = 21 - 3;

    /*
    int controller_read_position = 0;

    int vertical_scan = -1;
    int horizontal_scan = 21 - 3;
    int instruction_countdown = 1;

    int64_t ppu_cycle = 21 - 3;
    int64_t frame_count = 0;
    int64_t num_actions = 0;
    */

    int8u_t statusRegisterByteGet() {
        return (N << 7) | (O << 6) | (U << 5) | (B << 4) |
               (D << 3) | (I << 2) | (Z << 1) | (C << 0);
    }
};
#endif

/*
 int8u_t array[0] = {};
 int8u_t cpu_memory[0x800] = {};
 int8u_t ppu_registers[8] = {};
 int8u_t ppu_memory[0x4000] = {};
 int8u_t cartridge_memory[0x8000] = {};
 int8u_t ppu_OAM_register[1] = {};
 int8u_t control_port1[1] = {};
 int8u_t control_port2[1] = {};
 int8u_t ppu_OAM_memory[0x100] = {};
 int8u_t null_address_read[2] = {};
 int8u_t null_address_write[2] = {};
 */

#define CPU_MEMORY         0x0000
#define PPU_REGISTERS      (CPU_MEMORY + 0x800)
#define PPU_MEMORY         (PPU_REGISTERS + 8)
#define CARTRIDGE_MEMORY   (PPU_MEMORY + 0x4000)

#define NULL_ADDRESS_READ  0x10000
#define NULL_ADDRESS_WRITE (NULL_ADDRESS_READ + 2)

struct Memory {
    int8u_t cpu_memory[0x800];
    int8u_t ppu_registers[8];
    int8u_t ppu_memory[0x4000];
    int8u_t cartridge_memory[0x8000];
    // int8u_t null_address_read[2] = {};
    // int8u_t null_address_write[2] = {};
};

#ifdef __METAL__
int Memory__mapOffset(int index) {
    return
         (0x0000 <= index && index < 0x2000)*CPU_MEMORY +
         (0x8000 <= index && index < 0x10000)*CARTRIDGE_MEMORY;
}

int Memory__mapIndex(int offset, int index) {
    return (offset ==       CPU_MEMORY)*(index & 0x07FF) +
           (offset == CARTRIDGE_MEMORY)*(index - 0x8000);
}

/*
int mapOffset(int index) {
    return      // (CPU_MEMORY)*(0x0000 <= index && index < 0x2000) +
                (PPU_REGISTERS)*(0x2000 <= index && index < 0x4000) +
    (NULL_ADDRESS_WRITE_OFFSET)*(0x4000 <= index && index < 0x4014) +
             (PPU_OAM_REGISTER)*(0x4014 == index) +
    (NULL_ADDRESS_WRITE_OFFSET)*(0x4015 == index) +
               (CONTROL_PORT_1)*(0x4016 == index) +
               (CONTROL_PORT_2)*(0x4017 == index) +
    (NULL_ADDRESS_WRITE_OFFSET)*(0x4018 <= index && index < 0x8000) +
          // (CARTRIDGE_MEMORY)*(0x8000 <= index && index < 0x10000) +
     (NULL_ADDRESS_READ_OFFSET)*(NULL_ADDRESS_READ == index) +
    (NULL_ADDRESS_WRITE_OFFSET)*(NULL_ADDRESS_WRITE == index);
}

int mapIndex(int offset, int index) {
    return // (index & 0x7FF)*(CPU_MEMORY       == offset) +
       ((index - 0x2000) & 7)*(PPU_REGISTERS    == offset) +
                          (0)*(PPU_OAM_REGISTER == offset) +
                          (0)*(CONTROL_PORT_1   == offset) +
                          (0)*(CONTROL_PORT_2   == offset) +
                          (0)*(PPU_OAM_REGISTER == offset) +
          // (index - 0x8000)*(CARTRIDGE_MEMORY == offset) +
                          (0)*(NULL_ADDRESS_READ_OFFSET  == offset) +
                          (0)*(NULL_ADDRESS_WRITE_OFFSET == offset);
}
*/

int8u_t Memory__readMemoryLogical(device Memory& memory, int index) {
    int offset = Memory__mapOffset(index);
    index = Memory__mapIndex(offset, index);

    return memory.cpu_memory[offset + index];
}

void Memory__writeMemoryLogical(device Memory& memory, int index, int8u_t data) {
    int offset = Memory__mapOffset(index);
    index = Memory__mapIndex(offset, index);

    memory.cpu_memory[offset + index] = data;
}

int8u_t Memory__readMemoryRaw(device Memory& memory, int index) {
    int offset = Memory__mapOffset(index);
    index = Memory__mapIndex(offset, index);

    return memory.cpu_memory[offset + index];
}

void Memory__writeMemoryRaw(device Memory& memory, int index, int8u_t data) {
    int offset = Memory__mapOffset(index);
    index = Memory__mapIndex(offset, index);

    memory.cpu_memory[offset + index] = data;
}
#endif

#ifdef __SWIFT__
enum MemoryOffset {
    cpu_memory_offset = offsetof(struct Memory, cpu_memory),
    cpu_memory_size = 0x800,
    cartridge_memory_offset = offsetof(struct Memory, cartridge_memory),
    cartridge_memory_size = 0x8000,
};
#endif

struct SystemState {
    struct Memory global_memory;

    int16u_t program_counter_initial;
    int8u_t stack_offset_initial;
};

#ifdef __METAL__

#include "regions.h"
#include "instructions.h"

void scanlineNext(thread ComputationState* state, device Memory& memory) {
    state->horizontal_scan = (state->horizontal_scan + 1) % 341;
}

void SystemState__next(thread ComputationState* state, device Memory& memory, device Trace* traceLines, int traceLinesIndex) {
    state->opcode = Memory__readMemoryRaw(memory, state->program_counter + 0);
    state->data1  = Memory__readMemoryRaw(memory, state->program_counter + 1);
    state->data2  = Memory__readMemoryRaw(memory, state->program_counter + 2);

    scanlineNext(state, memory);
    scanlineNext(state, memory);
    scanlineNext(state, memory);

    traceLines[traceLinesIndex].program_counter = state->program_counter;
    traceLines[traceLinesIndex].opcode = state->opcode;
    traceLines[traceLinesIndex].byte1 = state->data1;
    traceLines[traceLinesIndex].byte2 = state->data2;
    traceLines[traceLinesIndex].A = state->A;
    traceLines[traceLinesIndex].X = state->X;
    traceLines[traceLinesIndex].Y = state->Y;
    traceLines[traceLinesIndex].status_register = state->statusRegisterByteGet();
    traceLines[traceLinesIndex].stack_offset = state->stack_offset;
    traceLines[traceLinesIndex].horizontal_scan = state->horizontal_scan;

    instructions[state->opcode].transition(state, memory);
}

#endif

/*
__device__
void next(ComputationState* state, Memory& memory) {
    int8u_t opcode = memory[state->program_counter + 0];
    state->data1   = memory[state->program_counter + 1];
    state->data2   = memory[state->program_counter + 2];

    scanlineNext(this, state, memory);
    state->ppu_cycle++;
    scanlineNext(this, state, memory);
    state->ppu_cycle++;
    scanlineNext(this, state, memory);
    state->ppu_cycle++;

    state->is_DMA_active = (state->is_DMA_active || (state->is_DMA_should_start && state->ppu_cycle % 2 == 1));

    // clear is_DMA_should_start as soon as is_DMA_active is set
    state->is_DMA_should_start = (state->is_DMA_should_start != (state->is_DMA_should_start && state->ppu_cycle % 2 == 1));

    state->instruction_countdown -= (!state->is_DMA_active);
    bool instruction_OK = state->instruction_countdown == 0;
    opcode = (NOP_instruction)*(!instruction_OK) + (opcode)*instruction_OK;

    bool nmi_condition = instruction_OK &&
                         (!state->has_vblank_nmi) &&
                         ((memory.ppu_registers[0x00] & 0x80) == 0x80) && // NMI enabled
                         ((state->ppu_status & 0x80) == 0x80); // vblank has occurred

    state->has_vblank_nmi |= nmi_condition;

    int8u_t nmi_L = memory[0xFFFA];
    int8u_t nmi_H = memory[0xFFFB];

    opcode = (opcode)*(!nmi_condition) + (NMI_instruction)*(nmi_condition);
    state->data1 = (state->data1)*(!nmi_condition) + (nmi_L)*nmi_condition;
    state->data2 = (state->data2)*(!nmi_condition) + (nmi_H)*nmi_condition;

    bool odd_cycle = ((state->is_DMA_active) && (state->ppu_cycle % 2 == 1));
    bool even_cycle = ((state->is_DMA_active) && (state->ppu_cycle % 2 == 0));
    opcode = (opcode)*(!odd_cycle) + (DMA_read1_instruction)*odd_cycle;
    opcode = (opcode)*(!even_cycle) + (DMA_write1_instruction)*even_cycle;

    state->data1 = (state->data1)*(!odd_cycle) + (state->DMA_source_L + state->DMA_index)*odd_cycle;
    state->data2 = (state->data2)*(!odd_cycle) + (state->DMA_source_H)*odd_cycle;
    state->data1 = (state->data1)*(!even_cycle) + (state->DMA_index)*even_cycle;

    state->DMA_index += even_cycle;
    bool new_is_DMA_active = (state->is_DMA_active && !(even_cycle && state->DMA_index == 0));
    state->is_OAM_memory_invalidated = (state->is_OAM_memory_invalidated) || (new_is_DMA_active != state->is_DMA_active);
    state->is_DMA_active = new_is_DMA_active;

    #ifdef DEBUG
    if (blockIdx.x == 0 && threadIdx.x == OBSERVED_INSTANCE && instruction_OK) {
        uint8_t* opcodes = &memory[state->program_counter];
        traceWrite(state->program_counter, opcodes, state);
        trace_lines[trace_lines_index] = traceLineLast;
        trace_lines_index++;
    }
    #endif

    state->opcode = opcode;
    operationTransition(opcode, this, state, memory);
}
*/
