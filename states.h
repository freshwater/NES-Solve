
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
    int8u_t data1;
    int8u_t data2;

    int8u_t value1;
    int8u_t value2;
    int8u_t value3;

    int16u_t address;

    int8u_t store;

    int16_t vertical_scan = 0;
    int16_t horizontal_scan = 21 - 3;
    int16u_t ppu_cycle = 21 - 3;
    int16u_t cycle = 7;
    int16u_t instruction_countdown = 1;

    int8u_t ppu_status = 0;
    int8u_t ppu_address_H;
    int8u_t ppu_address_L;
    bool ppu_address_latch = false;

    bool is_dma_active = false;
    int8u_t dma_count = 0;
    int8u_t dma_source_H;
    int8u_t dma_source_L;
    int8u_t dma_target_H;
    int8u_t dma_target_L;

    bool has_blanked = false;
    bool has_vblank_nmi = false;
};

#define PPU_OFFSET      0x4000
#define PPU_OAM_OFFSET  (0x8000 - 256)

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

struct Memory {
    int8u_t array[0x10000 + NULL_ADDRESS_MARGIN] = {};

    __device__
    int8u_t read(int index, ComputationState* computation_state) {
        int16u_t address =                   (index % 0x800)*(0x0000 <= index && index < 0x2000) |
                           (0x2000 + ((index - 0x2000) % 8))*(0x2000 <= index && index < 0x4000) |
                                                     (index)*(0x4000 <= index && index < 0x8000) |
                                                     (index)*(0x8000 <= index /*&& index < 0xFFFF*/);

        int8u_t value = array[address];

        array[address] &= (address == PPU_STATUS) ? 0b01111111 : 0xFF;
        computation_state->ppu_address_latch = (address == PPU_STATUS) ? false : computation_state->ppu_address_latch;

        return value;
    }

    __device__
    void write(int index, int8u_t value, ComputationState* computation_state) {
        int16u_t address =                   (index % 0x800)*(0x0000 <= index && index < 0x2000) |
                           (0x2000 + ((index - 0x2000) % 8))*(0x2000 <= index && index < 0x4000) |
                                                     (index)*(0x4000 <= index && index < 0x8000) |
                                                     (index)*(0x8000 <= index /*&& index < 0xFFFF*/);

        computation_state->ppu_address_H = (address == PPU_ADDRESS && computation_state->ppu_address_latch == false) ? value : computation_state->ppu_address_H;
        computation_state->ppu_address_L = (address == PPU_ADDRESS && computation_state->ppu_address_latch == true) ? value : computation_state->ppu_address_L;
        computation_state->ppu_address_latch = (address == PPU_ADDRESS) ? !(computation_state->ppu_address_latch) : computation_state->ppu_address_latch;

        /* map to PPU RAM if necessary */

        bool is_ppu_data_write = (address == PPU_DATA);
        int16u_t ppu_address = ((computation_state->ppu_address_H << 8) | computation_state->ppu_address_L);

        address = is_ppu_data_write ? PPU_OFFSET + ppu_address : address;
        array[address] = value;

        ppu_address += is_ppu_data_write;
        computation_state->ppu_address_H = ppu_address >> 8;
        computation_state->ppu_address_L = 0xFF & ppu_address;

        bool is_DMA_write = (address == PPU_OAM_DMA);
        computation_state->is_dma_active |= is_DMA_write;
        computation_state->dma_source_H = (computation_state->dma_source_H)*(!is_DMA_write) + (value)*is_DMA_write;
        computation_state->dma_source_L = (computation_state->dma_source_L)*(!is_DMA_write) + (0)*is_DMA_write;

        bool is_OAM_address_write = (address == PPU_OAM_ADDRESS);
        computation_state->dma_target_H = (computation_state->dma_target_H)*(!is_OAM_address_write) + (PPU_OAM_OFFSET >> 8)*is_OAM_address_write;
        computation_state->dma_target_L = (computation_state->dma_target_L)*(!is_OAM_address_write) + ((PPU_OAM_OFFSET & 0xFF) + value)*is_OAM_address_write;
    }

    __device__
    int8u_t& operator[](int index) {
        index =                   (index % 0x800)*(0x0000 <= index && index < 0x2000) |
                (0x2000 + ((index - 0x2000) % 8))*(0x2000 <= index && index < 0x4000) |
                                          (index)*(0x4000 <= index && index < 0x8000) |
                                          (index)*(0x8000 <= index /*&& index < 0xFFFF*/);

        return array[index];
    }
};

struct SystemState {
    int16u_t program_counter; 
    Memory memory;
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

    ComputationState computation_state;

    #ifdef DEBUG
    Trace traceLineData[100000];
    int traceIndex = 0;
    #endif

    SystemState(std::vector<char>& program, int16u_t program_counter, int load_point) {
        std::copy(program.begin(), program.end(), &memory.array[load_point]);
        this->program_counter = program_counter;
        this->stack_offset = 0xFD;
    }

    SystemState(std::vector<char>& program) {
        std::copy(program.begin(), program.end(), &memory.array[0x8000]);
        this->program_counter = (memory.array[0xFFFD] << 8) | memory.array[0xFFFC];
        this->stack_offset = 0xFD;
    }

    __device__
    uint8_t statusRegisterByteGet() {
        return (N << 7) | (O << 6) | (U << 5) | (B << 4) |
               (D << 3) | (I << 2) | (Z << 1) | (C << 0);
    }

    __device__
    void traceWrite(uint16_t program_counter, int8u_t* opcodes) {
        #ifdef DEBUG
        traceLineData[traceIndex] = {
            .program_counter = program_counter,
            .opcode = opcodes[0],
            .byte1 = opcodes[1],
            .byte2 = opcodes[2],
            .A = A,
            .X = X,
            .Y = Y,
            .status_register = statusRegisterByteGet(),
            .stack_offset = stack_offset,
            .vertical_scan = computation_state.vertical_scan,
            .horizontal_scan = computation_state.horizontal_scan,
            .cycle = (int16u_t)(computation_state.ppu_cycle / 3)
        };

        traceIndex++;
        #endif
    }

    __device__
    static void scanLineNext(SystemState* state, ComputationState* computation_state) {
        computation_state->horizontal_scan += 1;
        computation_state->vertical_scan += (computation_state->horizontal_scan >= 341);
        computation_state->horizontal_scan %= 341;
        computation_state->vertical_scan = (computation_state->vertical_scan == 261) ? -1 : computation_state->vertical_scan;

        bool blank_condition = !(computation_state->has_blanked) &&
                                (computation_state->vertical_scan == 241) &&
                                (computation_state->horizontal_scan == 2);

        state->memory[PPU_STATUS] |= blank_condition ? 0x80 : 0x00;
        computation_state->has_blanked = computation_state->has_blanked || blank_condition;
        computation_state->has_blanked = computation_state->has_blanked && (computation_state->vertical_scan > 0);
        computation_state->has_vblank_nmi = computation_state->has_vblank_nmi && (computation_state->vertical_scan > 0);
    }

    __device__
    void next() {
        int8u_t* opcodes = &memory.array[program_counter];

        computation_state.ppu_cycle++;
        scanLineNext(this, &computation_state);
        computation_state.ppu_cycle++;
        scanLineNext(this, &computation_state);
        computation_state.ppu_cycle++;
        scanLineNext(this, &computation_state);

        computation_state.data1 = 0xFF & opcodes[1];
        computation_state.data2 = 0xFF & opcodes[2];

        computation_state.instruction_countdown -= (!computation_state.is_dma_active);
        bool instruction_OK = computation_state.instruction_countdown == 0;
        int8u_t opcode = (0x64)*(!instruction_OK) + (opcodes[0])*instruction_OK;

        bool nmi_condition = instruction_OK &&
                             (!computation_state.has_vblank_nmi) &&
                             ((memory[PPU_CTRL1] & 0x80) == 0x80) && // NMI enabled
                             ((memory[PPU_STATUS] & 0x80) == 0x80); // vblank has occurred

        computation_state.has_vblank_nmi |= nmi_condition;

        int8u_t nmi_L = memory[0xFFFA];
        int8u_t nmi_H = memory[0xFFFB];

        opcode = (opcode)*(!nmi_condition) + (0x20)*(nmi_condition);
        computation_state.data1 = (computation_state.data1)*(!nmi_condition) + (nmi_L)*nmi_condition;
        computation_state.data2 = (computation_state.data2)*(!nmi_condition) + (nmi_H)*nmi_condition;

        bool odd_cycle = ((computation_state.is_dma_active) && (computation_state.ppu_cycle % 2 == 1));
        bool even_cycle = ((computation_state.is_dma_active) && (computation_state.ppu_cycle % 2 == 0));
        opcode = (opcode)*(!odd_cycle) + (DMA_read1)*odd_cycle;
        opcode = (opcode)*(!even_cycle) + (DMA_write1)*even_cycle;

        computation_state.data1 = (computation_state.data1)*(!odd_cycle) + (computation_state.dma_source_L + computation_state.dma_count)*odd_cycle;
        computation_state.data2 = (computation_state.data2)*(!odd_cycle) + (computation_state.dma_source_H)*odd_cycle;
        computation_state.data1 = (computation_state.data1)*(!even_cycle) + (computation_state.dma_target_L + computation_state.dma_count)*even_cycle;
        computation_state.data2 = (computation_state.data2)*(!even_cycle) + (computation_state.dma_target_H)*even_cycle;

        computation_state.dma_count += even_cycle;
        // computation_state.is_dma_active = (computation_state.is_dma_active && computation_state.dma_count != 0);
        computation_state.is_dma_active = (computation_state.is_dma_active && !(even_cycle && computation_state.dma_count == 0));

        if (instruction_OK) {
            traceWrite(program_counter, opcodes);
        }

        operationTransition(opcode, this, &computation_state);
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
