
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
    int16_t vertical_scan;
    int16_t horizontal_scan;
    int16u_t cycle;
};

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

    int8u_t store;
    int8u_t ppu_status;

    int controller_read_position = 0;

    int vertical_scan = -1;
    int horizontal_scan = 21 - 3;
    int instruction_countdown = 1;

    int64_t ppu_cycle = 21 - 3;
    int64_t frame_count = 0;
    int64_t num_actions = 0;

    struct Sprite {
        int8u_t Y;
        int8u_t tile_id;
        int8u_t attributes;
        int8u_t X;
    };

    Sprite sprites[64];
    bool is_OAM_memory_invalidated = true;

    uint32_t sprite_MSB_line[9];
    uint32_t sprite_LSB_line[9];
    uint32_t sprite_palette_bit1_line[9];
    uint32_t sprite_palette_bit0_line[9];
    uint32_t sprite_zero_line[9];
    bool sprite_zero_hit_possible;
    bool has_sprite_zero_hit;
    int sprites_intersecting_index = 0;

    int8u_t ppu_address_H;
    int8u_t ppu_address_L;
    int16u_t loopy_V;
    int16u_t loopy_T;
    int16u_t loopy_X;
    int16u_t background_sprite_MSB_shift_register;
    int16u_t background_sprite_LSB_shift_register;
    int16u_t background_palette_MSB_shift_register;
    int16u_t background_palette_LSB_shift_register;
    int16u_t nametable_next_tile_id;
    int16u_t background_next_attribute;
    int8u_t character_next_MSB_plane;
    int8u_t character_next_LSB_plane;

    bool ppu_address_latch = false;
    volatile int8u_t ppu_data_buffer;

    bool is_DMA_should_start = false;
    bool is_DMA_active = false;
    int8u_t DMA_index = 0;
    int8u_t DMA_source_H;
    int8u_t DMA_source_L;

    bool has_blanked = false;
    bool has_vblank_nmi = false;

    uint8_t control_port1;

    float kernel_sums[256];
    uint8_t kernel_sums_index = 0;
    int kernel_X = 0;
    int kernel_Y = 0;
    float kernel_sum = 0;

    __device__
    uint8_t statusRegisterByteGet() {
        return (N << 7) | (O << 6) | (U << 5) | (B << 4) |
               (D << 3) | (I << 2) | (Z << 1) | (C << 0);
    }

};

#define NOP_instruction         0x6B
#define NMI_instruction         0xC2
#define DMA_read1_instruction   0xAB
#define DMA_write1_instruction  0xB2

#define CPU_MEMORY         0x0000
#define PPU_REGISTERS      (CPU_MEMORY + 0x800)
#define PPU_MEMORY         (PPU_REGISTERS + 8)
#define CARTRIDGE_MEMORY   (PPU_MEMORY + 0x4000)
#define PPU_OAM_REGISTER   (CARTRIDGE_MEMORY + 0x8000)
#define CONTROL_PORT_1     (PPU_OAM_REGISTER + 1)
#define CONTROL_PORT_2     (CONTROL_PORT_1 + 1)
#define PPU_OAM_MEMORY     (CONTROL_PORT_2 + 1)
#define NULL_ADDRESS_READ_OFFSET (PPU_OAM_MEMORY + 0x100)
#define NULL_ADDRESS_WRITE_OFFSET (NULL_ADDRESS_READ_OFFSET + 2)

#define NULL_ADDRESS_READ  0x10000
#define NULL_ADDRESS_WRITE (NULL_ADDRESS_READ + 2)

struct Memory {
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

    __device__
    int mapOffset(int index) {
        return         (CPU_MEMORY)*(0x0000 <= index && index < 0x2000) +
                    (PPU_REGISTERS)*(0x2000 <= index && index < 0x4000) +
        (NULL_ADDRESS_WRITE_OFFSET)*(0x4000 <= index && index < 0x4014) +
                 (PPU_OAM_REGISTER)*(0x4014 == index) +
        (NULL_ADDRESS_WRITE_OFFSET)*(0x4015 == index) +
                   (CONTROL_PORT_1)*(0x4016 == index) +
                   (CONTROL_PORT_2)*(0x4017 == index) +
        (NULL_ADDRESS_WRITE_OFFSET)*(0x4018 <= index && index < 0x8000) +
                 (CARTRIDGE_MEMORY)*(0x8000 <= index && index < 0x10000) +
         (NULL_ADDRESS_READ_OFFSET)*(NULL_ADDRESS_READ == index) +
        (NULL_ADDRESS_WRITE_OFFSET)*(NULL_ADDRESS_WRITE == index);
    }

    __device__
    int mapIndex(int offset, int index) {
        return    (index & 0x7FF)*(CPU_MEMORY       == offset) +
           ((index - 0x2000) & 7)*(PPU_REGISTERS    == offset) +
                              (0)*(PPU_OAM_REGISTER == offset) +
                              (0)*(CONTROL_PORT_1   == offset) +
                              (0)*(CONTROL_PORT_2   == offset) +
                              (0)*(PPU_OAM_REGISTER == offset) +
                 (index - 0x8000)*(CARTRIDGE_MEMORY == offset) +
                              (0)*(NULL_ADDRESS_READ_OFFSET  == offset) +
                              (0)*(NULL_ADDRESS_WRITE_OFFSET == offset);
    }

    __device__
    int8u_t& ppuMemoryMapped(int index) {
        return ppu_memory[ppuMapAddress(index)];
    }

    __device__
    int8u_t ppuPaletteRead(int index) {
        index &= 0x1F;
        bool any_palette_background = ((index & 0x03) == 0);
        index = any_palette_background ? 0 : index;

        return ppu_memory[0x3F00 + index];
    }

    __device__
    int16u_t ppuMapAddress(int16u_t address) {
        bool is_palette_address = (0x3F00 <= address);
        address = is_palette_address ? (0x3F00 | (address & 0x001F)) : address;

        /*
        http://wiki.nesdev.com/w/index.php/PPU_palettes#Memory_Map
        Addresses $3F00/$3F04/$3F08/$3F0C are mirrored by $3F10/$3F14/$3F18/$3F1C. */
        /* Aside from the mirroring, the read/write works as normal for the CPU,
           but the PPU render circuit reads these from $3F00 (see ppuPaletteRead). */

        bool any_palette_mirror = (0x3F10 <= address && ((address & 0x03) == 0));
        bool is_nametable_mirror_region1 = (0x2800 <= address && address < 0x2C00);
        bool is_nametable_mirror_region2 = (0x2C00 <= address && address < 0x3000);

        address = is_nametable_mirror_region1 ? (address - 0x800) : address;
        address = is_nametable_mirror_region2 ? (address - 0x800) : address;

        address = any_palette_mirror ? (address - 0x0010) : address;

        return address;
    }

    __device__
    int8u_t read(int index0, ComputationState* state) {
        int offset = mapOffset(index0);
        int index = mapIndex(offset, index0);
        int16u_t address = offset + index;

        bool is_ppu_data_read = (offset == PPU_REGISTERS) && (index == 0x07);
        int16u_t ppu_address = ((state->ppu_address_H << 8) | state->ppu_address_L);
        address = is_ppu_data_read ? (PPU_MEMORY + ppuMapAddress(ppu_address)) : address;

        int value = array[address];

        if (is_ppu_data_read) {
            bool is_ppu_buffered_read = ppu_address < 0x3F00;
            int8u_t buffer_temp = state->ppu_data_buffer;
            state->ppu_data_buffer = value;
            value = is_ppu_buffered_read ? buffer_temp : value;

            int8u_t ppu_ctrl = ppu_registers[0x00];
            ppu_address += (31*((ppu_ctrl >> 2) & 0x01) + 1);
            state->ppu_address_H = ppu_address >> 8;
            state->ppu_address_L = 0xFF & ppu_address;
        }

        if ((offset == PPU_REGISTERS) && (index == 0x02)) {
            value = state->ppu_status;
            state->ppu_status &= 0b01111111;
            state->ppu_address_latch = false;
        }

        if (offset == CONTROL_PORT_1)
        {
            value = state->control_port1;
            value = ((value << state->controller_read_position) & 0x80) >> 7;

            if (state->controller_read_position > 7) {
                value = 0x01;
            }

            state->controller_read_position += 1;
        }

        return value;
    }

    __device__
    void write(int index0, int8u_t value, bool is_oam_dma_write, ComputationState* state) {
        int offset = mapOffset(index0);
        int index = mapIndex(offset, index0);

        bool is_ppu_address = (offset == PPU_REGISTERS) && (index == 0x06);
        state->ppu_address_H = (is_ppu_address && state->ppu_address_latch == false) ? value : state->ppu_address_H;
        state->ppu_address_L = (is_ppu_address && state->ppu_address_latch == true) ? value : state->ppu_address_L;

        // /* map to PPU RAM if necessary */

        bool is_ppu_data_write = (offset == PPU_REGISTERS) && (index == 0x07);
        int16u_t ppu_address = ((state->ppu_address_H << 8) | state->ppu_address_L);

        bool is_controller_write = (offset == CONTROL_PORT_1) && (index == 0x00);

        /* PPU data pre write */
        int address = is_ppu_data_write ? PPU_MEMORY + ppuMapAddress(ppu_address) : offset + index;
        address = is_oam_dma_write ? PPU_OAM_MEMORY + index0 : address;
        address = is_controller_write ? NULL_ADDRESS_WRITE_OFFSET : address;

        array[address] = value;

        /* PPU internal state */
        bool is_2000_write = (offset == PPU_REGISTERS) && (index == 0x00);
        bool is_2002_write = (offset == PPU_REGISTERS) && (index == 0x02);
        bool is_2005_write0 = (offset == PPU_REGISTERS) && (index == 0x05) && (state->ppu_address_latch == false);
        bool is_2005_write1 = (offset == PPU_REGISTERS) && (index == 0x05) && (state->ppu_address_latch == true);
        bool is_2006_write0 = (offset == PPU_REGISTERS) && (index == 0x06) && (state->ppu_address_latch == false);
        bool is_2006_write1 = (offset == PPU_REGISTERS) && (index == 0x06) && (state->ppu_address_latch == true);
        bool any_latch_write = (offset == PPU_REGISTERS) && ((index == 0x05) || (index == 0x06));

        state->ppu_status = (state->ppu_status)*(!is_2002_write) + value*is_2002_write;

        /* https://wiki.nesdev.com/w/index.php/PPU_scrolling#Register_controls */
        state->loopy_T = is_2000_write ?   ((state->loopy_T &  (       ~(0b00000000'00000011  << 10)))
                                                            |  ((value & 0b00000000'00000011) << 10)) : state->loopy_T; // nametable set
        state->loopy_T = is_2005_write0 ?  ((state->loopy_T &  (       ~(0b00000000'11111000  >>  3)))
                                                            |  ((value & 0b00000000'11111000) >>  3)) : state->loopy_T; // x scroll set
        state->loopy_X = is_2005_write0 ?  (                   ((value & 0b00000000'00000111) >>  0)) : state->loopy_X;
        state->loopy_T = is_2005_write1 ?  ((state->loopy_T & ~((        0b00000000'00000111  << 12) | (         0b00000000'11111000  << 2)))
                                                            |  ((value & 0b00000000'00000111) << 12) | ((value & 0b00000000'11111000) << 2)) : state->loopy_T; // y scroll set
        state->loopy_T = is_2006_write0 ? (((state->loopy_T &  (       ~(0b00000000'00111111  <<  8)))
                                                            |  ((value & 0b00000000'00111111) <<  8)) & 0b00111111'11111111) : state->loopy_T; // ppu address high set
        state->loopy_T = is_2006_write1 ?  ((state->loopy_T &  (       ~(0b00000000'11111111  <<  0)))
                                                            |  ((value & 0b00000000'11111111) <<  0)) : state->loopy_T; // ppu address low set
        state->loopy_V = is_2006_write1 ? (state->loopy_T) : state->loopy_V;
        state->ppu_address_latch = any_latch_write ? (!state->ppu_address_latch) : state->ppu_address_latch;

        /* PPU data post write */
        int8u_t ppu_ctrl = ppu_registers[0x00];
        ppu_address += (31*((ppu_ctrl >> 2) & 0x01) + 1)*(is_ppu_data_write);
        state->ppu_address_H = ppu_address >> 8;
        state->ppu_address_L = 0xFF & ppu_address;

        bool is_DMA_register_write = (offset == PPU_OAM_REGISTER) && (index == 0x00);
        state->is_DMA_should_start = (state->is_DMA_should_start || is_DMA_register_write);
        state->DMA_source_H = (state->DMA_source_H)*(!is_DMA_register_write) + (value)*is_DMA_register_write;
        state->DMA_source_L = (state->DMA_source_L)*(!is_DMA_register_write) + (0)*is_DMA_register_write;

        // bool is_OAM_address_write = (offset == PPU_REGISTERS) && (index == 0x03);
        // state->dma_target_H = (state->dma_target_H)*(!is_OAM_address_write) + (PPU_OAM_OFFSET >> 8)*is_OAM_address_write;
        // state->dma_target_L = (state->dma_target_L)*(!is_OAM_address_write) + ((PPU_OAM_OFFSET & 0xFF) + value)*is_OAM_address_write;

        if (is_controller_write) {
            state->controller_read_position = 0;
        }
    }

    __device__
    int8u_t& operator[](int index) {
        int offset = mapOffset(index);
        index = mapIndex(offset, index);

        return array[offset + index];
    }
};

__device__
__constant__
const uint8_t colors[256*3] = {
    84, 84, 84, 0, 30, 116, 8, 16, 144, 48, 0, 136, 68, 0, 100, 92, 0, 48, 84, 4, 0, 60, 24, 0, 32, 42, 0, 8, 58, 0, 0, 64, 0, 0, 60, 0, 0, 50, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    152, 150, 152, 8, 76, 196, 48, 50, 236, 92, 30, 228, 136, 20, 176, 160, 20, 100, 152, 34, 32, 120, 60, 0, 84, 90, 0, 40, 114, 0, 8, 124, 0, 0, 118, 40, 0, 102, 120, 0, 0, 0,0, 0, 0, 0, 0, 0,
    236, 238, 236, 76, 154, 236, 120, 124, 236, 176, 98, 236, 228, 84, 236, 236, 88, 180, 236, 106, 100, 212, 136, 32, 160, 170, 0, 116, 196, 0, 76, 208, 32, 56, 204, 108, 56, 180, 204, 60, 60, 60,0, 0, 0, 0, 0, 0,
    236, 238, 236, 168, 204, 236, 188, 188, 236, 212, 178, 236, 236, 174, 236, 236, 174, 212, 236, 180, 176, 228, 196, 144, 204, 210, 120, 180, 222, 120, 168, 226, 144, 152, 226, 180, 160, 214, 228, 160, 162, 160, 0, 0, 0, 0, 0, 0};

__device__
const float kernel[16*16*3] = {
    #include "_asymmetric_kernel.h"
};

__device__
void operationTransition(uint8_t, SystemState*, ComputationState*, Memory&);

struct SystemState {
    int16u_t program_counter_initial;
    Memory global_memory;
    int8u_t stack_offset_initial;

    uint8_t* frames_red;
    uint8_t* frames_green;
    uint8_t* frames_blue;
    int64_t frames_pixel_index = 0;
    float* data_lines;
    int64_t data_lines_index = 0;

    uint8_t* hash_sets;

    Trace traceLineLast;
    Trace* trace_lines;
    int trace_lines_index = 0;

    #include "rendering_and_export.h"

    SystemState(std::vector<char>& program, int16u_t program_counter, int load_point) {
        std::copy(program.begin(), program.end(), global_memory.cartridge_memory);
        std::copy(program.begin(), program.end(), (global_memory.cartridge_memory + 0x4000));
        this->program_counter_initial = program_counter;
        this->stack_offset_initial = 0xFD;
    }

    SystemState(std::vector<char>& program_data, std::vector<char>& character_data) {
        std::copy(program_data.begin(), program_data.end(), global_memory.cartridge_memory);
        if (program_data.size() < 0x8000) {
            std::copy(program_data.begin(), program_data.end(), (global_memory.cartridge_memory + 0x4000));
        }
        std::copy(character_data.begin(), character_data.end(), global_memory.ppu_memory);
        this->program_counter_initial = (global_memory.cartridge_memory[0xFFFD % 0x8000] << 8) | global_memory.cartridge_memory[0xFFFC % 0x8000];
        this->stack_offset_initial = 0xFD;
    }

    __device__
    void traceWrite(uint16_t program_counter, int8u_t* opcodes, ComputationState* state) {
        traceLineLast = {
            .program_counter = program_counter,
            .opcode = opcodes[0],
            .byte1 = opcodes[1],
            .byte2 = opcodes[2],
            .A = state->A,
            .X = state->X,
            .Y = state->Y,
            .status_register = state->statusRegisterByteGet(),
            .stack_offset = state->stack_offset,
            .vertical_scan = state->vertical_scan,
            .horizontal_scan = state->horizontal_scan,
            .cycle = (int16u_t)(state->ppu_cycle / 3)
        };
    }

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
};

struct OperationInformation {
    std::string name = "NOP"; 
    int byte_count = 0;
    std::string format_type = "None";

    std::string doFormat(int8u_t byte1, int8u_t byte2, int16u_t program_counter) {
        char cs[10];

        if (format_type == "Absolute" || format_type == "AbsoluteDereference") {
            sprintf(cs, "$%04X ", (((uint8_t)byte2 << 8) | (uint8_t)byte1) & 0xFFFF);
        } else if (format_type == "AbsoluteAddressDereference") {
            sprintf(cs, "($%04X) ", (((uint8_t)byte2 << 8) | (uint8_t)byte1) & 0xFFFF);
        } else if (format_type == "AbsoluteX") {
            sprintf(cs, "$%04X,X ", (((uint8_t)byte2 << 8) | (uint8_t)byte1) & 0xFFFF);
        } else if (format_type == "AbsoluteY") {
            sprintf(cs, "$%04X,Y ", (((uint8_t)byte2 << 8) | (uint8_t)byte1) & 0xFFFF);
        } else if (format_type == "Immediate") {
            sprintf(cs, "#$%02X ", (uint8_t)byte1);
        } else if (format_type == "Zeropage" || format_type == "ZeropageDereference") {
            sprintf(cs, "$%02X ", (uint8_t)byte1);
        } else if (format_type == "IndirectX") {
            sprintf(cs, "($%02X,X) ", (uint8_t)byte1);
        } else if (format_type == "ZeropageX") {
            sprintf(cs, "$%02X,X ", (uint8_t)byte1);
        } else if (format_type == "ZeropageY") {
            sprintf(cs, "$%02X,Y ", (uint8_t)byte1);
        } else if (format_type == "IndirectY") {
            sprintf(cs, "($%02X),Y ", (uint8_t)byte1);
        } else if (format_type == "Implied") {
            strcpy(cs, "");
        } else if (format_type == "Address_Relative") {
            sprintf(cs, "$%04X ", program_counter + 2 + (int8_t)byte1);
        } else {
            return format_type;
        }

        return std::string(cs);
    }
};
