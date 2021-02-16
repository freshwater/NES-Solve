
struct Region_Wire {
    flag_t value1_from_data1 = 0;
    flag_t value1_from_X = 0;
    flag_t address_from_absolute = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        computation_state->value1 = ((computation_state->data1)*value1_from_data1 +
                                     (state->X)*value1_from_X);

        computation_state->address = ((computation_state->data2 << 8) | computation_state->data1)*address_from_absolute;
    }
};

struct Region_Rewire {
    flag_t X_from_value1 = 0;
    flag_t program_counter_from_address = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        state->X = (state->X)*(1-X_from_value1) + (computation_state->value1)*X_from_value1;

        state->program_counter = (state->program_counter)*(1-program_counter_from_address) + (computation_state->address)*program_counter_from_address;
    }
};

struct Region_Write {
    flag_t address_write = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        int16u_t address = (NULL_ADDRESS_WRITE)*(1-address_write) + (computation_state->address)*address_write;
        state->memory[address] = computation_state->value1;
    }
};

struct Region_ProgramCounter {
    const int_t PC_increment = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        state->program_counter += PC_increment;
    }
};

struct RegionComposition {
    const Region_Wire wire;
    const Region_Rewire rewire;
    const Region_Write write;
    const Region_ProgramCounter program_counter;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        wire.transition(state, computation_state);
        rewire.transition(state, computation_state);
        write.transition(state, computation_state);
        program_counter.transition(state, computation_state);
    }
};
