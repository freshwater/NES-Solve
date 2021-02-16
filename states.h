
struct Trace {
    int16u_t ProgramCounter;
    int8u_t opcode;
    int8u_t byte1;
    int8u_t byte2;
    int8u_t A;
    int8u_t X;
    int8u_t Y;
};

struct ComputationState {
    int8u_t data1;
    int8u_t data2;

    int8u_t value1;
    int16u_t address;
};

struct SystemState {
    int8u_t program_counter; 
    int8u_t memory[0x10000 + NULL_ADDRESS_MARGIN] = {};
    // uint8u_memory_t memory[0x10000 + NULL_ADDRESS_MARGIN] = {};

    int8u_t A = 0;
    int8u_t X = 0;
    int8u_t Y = 0;

    ComputationState computation_state;
    Trace traceLineData[100];
    int traceIndex = 0;

    SystemState(std::vector<char>& program, int program_counter, int load_point) {
        std::copy(program.begin(), program.end(), &memory[load_point]);
        this->program_counter = program_counter;
    }

    __device__
    void traceWrite(uint16_t program_counter, int8u_t* opcodes) {
        traceLineData[traceIndex] = {
            .ProgramCounter = program_counter,
            .opcode = opcodes[0],
            .byte1 = opcodes[1],
            .byte2 = opcodes[2],
            .A = A,
            .X = X,
            .Y = Y
        };

        traceIndex++;
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

    std::string doFormat(int8u_t byte1, int8u_t byte2) {
        char cs[10];

        if (format_type == "Absolute") {
            sprintf(cs, "$%04X", (((uint8_t)byte2 << 8) | (uint8_t)byte1) & 0xFFFF);
        } else if (format_type == "Immediate") {
            sprintf(cs, "#$%02X", byte1);
        } else if (format_type == "Zeropage") {
            sprintf(cs, "$%02X", byte1);
        } else {
            return format_type;
        }

        return std::string(cs);
    }
};
