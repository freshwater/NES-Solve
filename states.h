
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

    __device__
    uint8_t statusRegisterByteGet() {
        return (N << 7) | (O << 6) | (U << 5) | (B << 4) |
               (D << 3) | (I << 2) | (Z << 1) | (C << 0);
    }

    int8u_t data1;
    int8u_t data2;

    int8u_t value1;
    int8u_t value2;
    int8u_t value3;

    int16u_t address;

    int8u_t store;

    int8u_t opcode;

    int8u_t controller_read_position = 0;
    int frame_count = 0;

    int16_t vertical_scan = 0;
    int16_t horizontal_scan = 21 - 3;
    int16u_t ppu_cycle = 21 - 3;
    int16u_t instruction_countdown = 1;

    int8u_t ppu_status = 0;
    int8u_t ppu_address_H;
    int8u_t ppu_address_L;
    bool ppu_address_latch = false;

    bool is_dma_active = false;
    int8u_t dma_count = 0;
    int8u_t dma_source_H;
    int8u_t dma_source_L;

    bool has_blanked = false;
    bool has_vblank_nmi = false;
};

#define PPU_CTRL1       0x2000
#define PPU_CTRL2       0x2000
#define PPU_STATUS      0x2002
#define PPU_OAM_ADDRESS 0x2003
#define PPU_OAM_DATA    0x2004
#define PPU_SCROLL      0x2005
#define PPU_ADDRESS     0x2006
#define PPU_DATA        0x2007
#define PPU_OAM_DMA     0x4014

#define DMA_read1       0xAB
#define DMA_write1      0xB2

#define CPU_MEMORY         0x0000
#define PPU_REGISTERS      (CPU_MEMORY + 0x800)
#define PPU_MEMORY         (PPU_REGISTERS + 8)
#define CARTRIDGE_MEMORY   (PPU_MEMORY + 0x4000)
#define PPU_OAM_REGISTER   (CARTRIDGE_MEMORY + 0x8000)
#define CONTROL_PORT_1     (PPU_OAM_REGISTER + 1)
#define PPU_OAM_MEMORY     (CONTROL_PORT_1 + 1)
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
    int8u_t ppu_OAM_memory[0x100] = {};
    int8u_t null_address_read[2] = {};
    int8u_t null_address_write[2] = {};

    __device__
    int map_offset(int index) {
        return        (CPU_MEMORY)*(0x0000 <= index && index < 0x2000) +
                   (PPU_REGISTERS)*(0x2000 <= index && index < 0x4000) +
                               (0)*(0x4000 <= index && index < 0x4014) +
                (PPU_OAM_REGISTER)*(0x4014 == index) +
                               (0)*(0x4015 == index) +
                  (CONTROL_PORT_1)*(0x4016 == index) +
                               (0)*(0x4014  < index && index < 0x8000) +
                (CARTRIDGE_MEMORY)*(0x8000 <= index && index < 0x10000) +
       (null_address_write - array)*(NULL_ADDRESS_WRITE == index);
    }

    __device__
    int map_index(int offset, int index) {
        return        (index % 0x800)*(CPU_MEMORY       == offset) +
               ((index - 0x2000) % 8)*(PPU_REGISTERS    == offset) +
                                  (0)*(PPU_OAM_REGISTER == offset) +
                     (index - 0x8000)*(CARTRIDGE_MEMORY == offset);
    }

    __device__
    int8u_t read(int index0, ComputationState* state) {
        int offset = map_offset(index0);
        int index = map_index(offset, index0);

        bool is_ppu_data_read = (offset == PPU_REGISTERS) && (index == 0x07);
        int16u_t ppu_address = ((state->ppu_address_H << 8) | state->ppu_address_L);
        int16u_t address = offset + index;
        address = (address)*(!is_ppu_data_read) + (PPU_MEMORY + palette_mirror_address(ppu_address))*is_ppu_data_read;

        int8u_t value = array[address];

        bool is_ppu_status = (offset == PPU_REGISTERS) && (index == 0x02);
        array[offset + index] &= is_ppu_status ? 0b01111111 : 0xFF;
        state->ppu_address_latch = is_ppu_status ? false : state->ppu_address_latch;

        bool is_controller_read = (offset == CONTROL_PORT_1) && (index == 0x00);
        if (is_controller_read)
        {
            value = ((value << state->controller_read_position) & 0x80) >> 7;

            if (state->controller_read_position > 7) {
                value = 0x01;
            }

            state->controller_read_position += 1;
        }

        return value;
    }

    __device__
    int16u_t palette_mirror_address(int16u_t ppu_address) {
        /*
        http://wiki.nesdev.com/w/index.php/PPU_palettes#Memory_Map
        Addresses $3F10/$3F14/$3F18/$3F1C are mirrors of
                  $3F00/$3F04/$3F08/$3F0C */
        bool any_ = (ppu_address == 0x3F10) ||
                    (ppu_address == 0x3F14) ||
                    (ppu_address == 0x3F18) ||
                    (ppu_address == 0x3F1C);

        return (ppu_address)*(!any_) +
               (0x3F00)*(ppu_address == 0x3F10) +
               (0x3F04)*(ppu_address == 0x3F14) +
               (0x3F08)*(ppu_address == 0x3F18) +
               (0x3F0C)*(ppu_address == 0x3F1C);
    }

    __device__
    void write(int index0, int8u_t value, bool is_oam_dma_write, ComputationState* state) {
        int offset = map_offset(index0);
        int index = map_index(offset, index0);

        bool is_ppu_address = (offset == PPU_REGISTERS) && (index == 0x06);
        state->ppu_address_H = (is_ppu_address && state->ppu_address_latch == false) ? value : state->ppu_address_H;
        state->ppu_address_L = (is_ppu_address && state->ppu_address_latch == true) ? value : state->ppu_address_L;
        state->ppu_address_latch = is_ppu_address ? !(state->ppu_address_latch) : state->ppu_address_latch;

        // /* map to PPU RAM if necessary */

        bool is_ppu_data_write = (offset == PPU_REGISTERS) && (index == 0x07);
        int16u_t ppu_address = ((state->ppu_address_H << 8) | state->ppu_address_L);

        bool is_controller_write = (offset == CONTROL_PORT_1) && (index == 0x00);

        int address = is_ppu_data_write ? PPU_MEMORY + ppu_address : offset + index;
        address = is_oam_dma_write ? PPU_OAM_MEMORY + index0 : address;
        address = is_controller_write ? NULL_ADDRESS_WRITE : address;
        address = is_ppu_data_write ? PPU_MEMORY + palette_mirror_address(ppu_address) : address;

        array[address] = value;

        ppu_address += is_ppu_data_write;
        state->ppu_address_H = ppu_address >> 8;
        state->ppu_address_L = 0xFF & ppu_address;

        bool is_DMA_register_write = (offset == PPU_OAM_REGISTER) && (index == 0x00);
        state->is_dma_active |= is_DMA_register_write;
        state->dma_source_H = (state->dma_source_H)*(!is_DMA_register_write) + (value)*is_DMA_register_write;
        state->dma_source_L = (state->dma_source_L)*(!is_DMA_register_write) + (0)*is_DMA_register_write;

        // bool is_OAM_address_write = (offset == PPU_REGISTERS) && (index == 0x03);
        // state->dma_target_H = (state->dma_target_H)*(!is_OAM_address_write) + (PPU_OAM_OFFSET >> 8)*is_OAM_address_write;
        // state->dma_target_L = (state->dma_target_L)*(!is_OAM_address_write) + ((PPU_OAM_OFFSET & 0xFF) + value)*is_OAM_address_write;

        state->controller_read_position = (state->controller_read_position)*(!is_controller_write) + (0)*is_controller_write;
    }

    __device__
    int8u_t& operator[](int index) {
        int offset = map_offset(index);
        index = map_index(offset, index);

        return array[offset + index];
    }
};

struct SystemState {
    int16u_t program_counter_initial;
    Memory memory;
    int8u_t stack_offset_initial;

    Trace traceLineLast;
    Trace* trace_lines;
    int trace_lines_index = 0;

    SystemState(std::vector<char>& program, int16u_t program_counter, int load_point) {
        // std::copy(program.begin(), program.end(), memory.cartridge_memory);
        std::copy(program.begin(), program.end(), (memory.cartridge_memory + 0x4000));
        // std::copy(character_data.begin(), character_data.end(), memory.ppu_memory);
        this->program_counter_initial = program_counter;
        this->stack_offset_initial = 0xFD;
    }

    SystemState(std::vector<char>& program_data, std::vector<char>& character_data) {
        std::copy(program_data.begin(), program_data.end(), memory.cartridge_memory);
        if (program_data.size() < 0x8000) {
            std::copy(program_data.begin(), program_data.end(), (memory.cartridge_memory + 0x4000));
        }
        std::copy(character_data.begin(), character_data.end(), memory.ppu_memory);
        this->program_counter_initial = (memory.cartridge_memory[0xFFFD % 0x8000] << 8) | memory.cartridge_memory[0xFFFC % 0x8000];
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
    static void scanLineNext(SystemState* system, ComputationState* state) {
        state->horizontal_scan += 1;
        state->vertical_scan += (state->horizontal_scan >= 341);
        state->horizontal_scan %= 341;
        state->vertical_scan = (state->vertical_scan == 261) ? -1 : state->vertical_scan;

        bool blank_condition = !(state->has_blanked) &&
                                (state->vertical_scan == 241) &&
                                (state->horizontal_scan == 2);

        system->memory[PPU_STATUS] |= blank_condition ? 0x80 : 0x00;
        state->has_blanked = state->has_blanked || blank_condition;
        state->has_blanked = state->has_blanked && (state->vertical_scan > 0);
        state->has_vblank_nmi = state->has_vblank_nmi && (state->vertical_scan > 0);

        state->frame_count += (state->vertical_scan == -1) && (state->horizontal_scan == 0);
    }

    int stack_pointer = 0;
    int count = 0;
    int count1 = 0;

    __device__
    void next(ComputationState* state) {
        int8u_t* opcodes = &memory[state->program_counter];

        state->ppu_cycle++;
        scanLineNext(this, state);
        state->ppu_cycle++;
        scanLineNext(this, state);
        state->ppu_cycle++;
        scanLineNext(this, state);

        state->data1 = 0xFF & opcodes[1];
        state->data2 = 0xFF & opcodes[2];

        state->instruction_countdown -= (!state->is_dma_active);
        bool instruction_OK = state->instruction_countdown == 0;
        int8u_t opcode = (0x6B)*(!instruction_OK) + (opcodes[0])*instruction_OK;

        bool nmi_condition = instruction_OK &&
                             (!state->has_vblank_nmi) &&
                             ((memory[PPU_CTRL1] & 0x80) == 0x80) && // NMI enabled
                             ((memory[PPU_STATUS] & 0x80) == 0x80); // vblank has occurred

        state->has_vblank_nmi |= nmi_condition;

        int8u_t nmi_L = memory[0xFFFA];
        int8u_t nmi_H = memory[0xFFFB];

        opcode = (opcode)*(!nmi_condition) + (0xC2)*(nmi_condition);
        state->data1 = (state->data1)*(!nmi_condition) + (nmi_L)*nmi_condition;
        state->data2 = (state->data2)*(!nmi_condition) + (nmi_H)*nmi_condition;

        bool odd_cycle = ((state->is_dma_active) && (state->ppu_cycle % 2 == 1));
        bool even_cycle = ((state->is_dma_active) && (state->ppu_cycle % 2 == 0));
        opcode = (opcode)*(!odd_cycle) + (DMA_read1)*odd_cycle;
        opcode = (opcode)*(!even_cycle) + (DMA_write1)*even_cycle;

        state->data1 = (state->data1)*(!odd_cycle) + (state->dma_source_L + state->dma_count)*odd_cycle;
        state->data2 = (state->data2)*(!odd_cycle) + (state->dma_source_H)*odd_cycle;
        state->data1 = (state->data1)*(!even_cycle) + (state->dma_count)*even_cycle;

        state->dma_count += even_cycle;
        state->is_dma_active = (state->is_dma_active && !(even_cycle && state->dma_count == 0));

        count++;

        #ifdef DEBUG
        if (threadIdx.x == 7 && instruction_OK) {
            traceWrite(state->program_counter, opcodes, state);
            trace_lines[trace_lines_index] = traceLineLast;
            trace_lines_index++;
        }
        #endif

        state->opcode = opcode;
        operationTransition(opcode, this, state);
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
