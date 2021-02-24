
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

    int16_t vertical_scan = 0;
    int16_t horizontal_scan = 21;
    int16u_t cycle = 7;

    int8u_t ppu_status = 0;
    int8u_t ppu_address_H;
    int8u_t ppu_address_L;
    bool ppu_address_latch = false;

    bool has_blanked = false;
};

#define PPU_STATUS 0x2002
#define PPU_ADDRESS 0x2006
#define PPU_DATA 0x2007
#define PPU_OFFSET 0x4000

struct Memory {
    int8u_t array[0x10000 + NULL_ADDRESS_MARGIN] = {};

    __device__
    int8u_t read(int index, ComputationState* computation_state) {
        int16u_t address =                   (index % 0x800)*(0x0000 <= index && index < 0x2000) |
                           (0x2000 + ((index - 0x2000) % 8))*(0x2000 <= index && index < 0x4000) |
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
    }

    __device__
    int8u_t& operator[](int index) {
        index =                   (index % 0x800)*(0x0000 <= index && index < 0x2000) |
                (0x2000 + ((index - 0x2000) % 8))*(0x2000 <= index && index < 0x4000) |
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
            .cycle = computation_state.cycle
        };

        traceIndex++;
        #endif
    }

    int count = 0;
    __device__
    void next() {
        int8u_t* opcodes = &memory.array[program_counter];

        if (count >= 0) {
            traceWrite(program_counter, opcodes);
        }

        count++;

        computation_state.data1 = 0xFF & opcodes[1];
        computation_state.data2 = 0xFF & opcodes[2];

        operationTransition(opcodes[0], this, &computation_state);
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
