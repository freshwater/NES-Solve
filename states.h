
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
};

struct ComputationState {
    int8u_t data1;
    int8u_t data2;

    int8u_t value1;
    int8u_t value2;
    int8u_t value3;
    int16u_t address;
};

struct SystemState {
    int8u_t program_counter; 
    int8u_t memory[0x10000 + NULL_ADDRESS_MARGIN] = {};
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
    Trace traceLineData[0x300];
    int traceIndex = 0;
    #endif

    SystemState(std::vector<char>& program, int program_counter, int load_point) {
        std::copy(program.begin(), program.end(), &memory[load_point]);
        this->program_counter = program_counter;
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
            .stack_offset = stack_offset
        };

        traceIndex++;
        #endif
    }

    __device__
    void next() {
        int8u_t* opcodes = &memory[program_counter];

        traceWrite(program_counter, opcodes);

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

        if (format_type == "Absolute") {
            sprintf(cs, "$%04X ", (((uint8_t)byte2 << 8) | (uint8_t)byte1) & 0xFFFF);
        } else if (format_type == "Immediate") {
            sprintf(cs, "#$%02X ", (uint8_t)byte1);
        } else if (format_type == "Zeropage" | format_type == "ZeropageDereference") {
            sprintf(cs, "$%02X ", (uint8_t)byte1);
        } else if (format_type == "Implied") {
            strcpy(cs, "");
        } else if (format_type == "Address_Relative") {
            sprintf(cs, "$%04X ", program_counter + 2 + (0xFF & byte1));
        } else {
            return format_type;
        }

        return std::string(cs);
    }
};
