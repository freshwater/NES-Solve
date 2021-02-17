
struct Region_Wire {
    flag_t value1_from_data1 = 0;
    flag_t value1_from_zeropage_dereference = 0;
    flag_t value1_from_X = 0;

    flag_t address_from_absolute = 0;
    flag_t address_from_zeropage = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        // return state.memory[(ComputationState.NULL_ADDRESS)*(1-condition) + (address)*condition]*condition
        int16u_t zeropage_address = (NULL_ADDRESS_READ)*(1-value1_from_zeropage_dereference) + (computation_state->data1)*value1_from_zeropage_dereference;
        computation_state->value1 = ((computation_state->data1)*value1_from_data1 +
                                     (state->memory[zeropage_address])*value1_from_zeropage_dereference +
                                     (state->X)*value1_from_X);

        computation_state->address = (((computation_state->data2 << 8) | computation_state->data1)*address_from_absolute +
                                      (computation_state->data1)*address_from_zeropage);
    }
};

struct Region_BooleanLogic {
    flag_t A_AND_value1 = 0;
    flag_t value3_output = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        int8u_t result = (state->A & computation_state->value1)*A_AND_value1;

        computation_state->value3 = (computation_state->value3)*(1-value3_output) + (result)*value3_output;
    }
};

struct Region_JSR_RTS_RTI {
    flag_t jsr_OK = 0;
    flag_t rts_OK = 0;
    flag_t rti_OK = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        int16u_t program_counter = state->program_counter + (-1)*(int)jsr_OK;
        int8u_t pc_H = program_counter >> 8;
        int8u_t pc_L = program_counter & 0x00FF;

        flag_t any_rt = rts_OK + rti_OK;
        int_t pre_offset = rts_OK + rti_OK;
        int_t post_offset = -jsr_OK;

        state->stack_offset += rti_OK;

        int16u_t read_address = (NULL_ADDRESS_READ)*(1-rti_OK) + (STACK_ZERO + state->stack_offset)*rti_OK;
        // int8u_t new_status = (state->memory[read_address])*rti_OK;
        // new_status = behaviors.Behaviors.read_special_status_bits_on_pull(state, new_status, is_PLP_or_RTI=self.rti_OK)
        // new_status = (state.status_register_byte())*(1-self.rti_OK) + (new_status)*self.rti_OK
        // state.status_register_byte_set(new_status)

        state->stack_offset += pre_offset;

        int16u_t write_address = (NULL_ADDRESS_WRITE)*(1-jsr_OK) + (STACK_ZERO + state->stack_offset)*jsr_OK;
        read_address = (NULL_ADDRESS_READ)*(1-any_rt) + (STACK_ZERO + state->stack_offset)*any_rt;
        state->memory[write_address] = pc_H;
        pc_L = (pc_L)*(1-any_rt) + (state->memory[read_address])*any_rt;

        state->stack_offset += pre_offset;
        state->stack_offset += post_offset;

        write_address = (NULL_ADDRESS_WRITE)*(1-jsr_OK) + (STACK_ZERO + state->stack_offset)*jsr_OK;
        read_address = (NULL_ADDRESS_READ)*(1-any_rt) + (STACK_ZERO + state->stack_offset)*any_rt;
        state->memory[write_address] = pc_L;
        pc_H = (pc_H)*(1-any_rt) + (state->memory[read_address])*any_rt;

        state->stack_offset += post_offset;

        flag_t any_ = jsr_OK + rts_OK + rti_OK;
        int16u_t new_program_counter = (program_counter)*(1-any_) + (computation_state->address)*jsr_OK + ((pc_H << 8) + pc_L)*any_rt;
        // new_program_counter = (program_counter)*(1-jsr_OK) + (computation_state->address)*jsr_OK;
        new_program_counter = new_program_counter + (1 + 3)*rts_OK;

        state->program_counter = (state->program_counter)*(1-any_) + (new_program_counter)*any_;
    }
};

struct Region_Rewire {
    flag_t value1_from_A = 0;
    flag_t value2_from_value1_bit6 = 0;

    flag_t A_from_value1 = 0;
    flag_t X_from_value1 = 0;

    flag_t program_counter_from_address = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        state->A = (state->A)*(1-A_from_value1) + (computation_state->value1)*A_from_value1;
        state->X = (state->X)*(1-X_from_value1) + (computation_state->value1)*X_from_value1;

        computation_state->value1 = (computation_state->value1)*(1-value1_from_A) + (state->A)*value1_from_A;
        computation_state->value2 = (computation_state->value2)*(1-value2_from_value1_bit6) + ((computation_state->value1 >> 6) & 0x01)*value2_from_value1_bit6;
        state->program_counter = (state->program_counter)*(1-program_counter_from_address) + (computation_state->address)*program_counter_from_address;
    }
};

struct Region_Branch {
    flag_t flag_match = 0;
    flag_t N_flag_branch = 0;
    flag_t O_flag_branch = 0;
    flag_t Z_flag_branch = 0;
    flag_t C_flag_branch = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        flag_t condition = ((state->N == flag_match)*N_flag_branch +
                            (state->O == flag_match)*O_flag_branch +
                            (state->Z == flag_match)*Z_flag_branch +
                            (state->C == flag_match)*C_flag_branch);

        state->program_counter = state->program_counter + (computation_state->value1)*condition;
    }
};

struct Region_Flags {
    flag_t N_keep = 0;
    int_t  N_adjust = 0;
    flag_t N_adjust_source = 0;
    flag_t O_keep = 0;
    int_t  O_adjust = 0;
    flag_t O_adjust_direct = 0;
    flag_t D_keep = 0;
    int_t  D_adjust = 0;
    flag_t I_keep = 0;
    int_t  I_adjust = 0;
    flag_t Z_keep = 0;
    int_t  Z_adjust = 0;
    flag_t Z_adjust_source = 0;
    flag_t C_keep = 0;
    int_t  C_adjust = 0;
    flag_t C_adjust_direct = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        int8u_t N_value = ((N_adjust_source == 1)*computation_state->value1 +
                           (N_adjust_source == 2)*computation_state->value2 +
                           (N_adjust_source == 3)*computation_state->value3);

        int8u_t O_direct_value = ((O_adjust_direct == 1)*computation_state->value1 +
                                  (O_adjust_direct == 2)*computation_state->value2 +
                                  (O_adjust_direct == 3)*computation_state->value3);

        int8u_t Z_value = ((Z_adjust_source == 1)*computation_state->value1 +
                           (Z_adjust_source == 2)*computation_state->value2 +
                           (Z_adjust_source == 3)*computation_state->value3);

        int8u_t C_direct_value = ((C_adjust_direct == 1)*computation_state->value1 +
                                  (C_adjust_direct == 2)*computation_state->value2 +
                                  (C_adjust_direct == 3)*computation_state->value3);

        state->N = (state->N)*N_keep + N_adjust + (N_value >> 7)*(N_adjust_source != 0);
        state->O = (state->O)*O_keep + O_adjust + (O_direct_value)*(O_adjust_direct != 0);
        state->D = (state->D)*D_keep + D_adjust;
        // state->I = (state->I)*I_keep + I_adjust;
        state->Z = (state->Z)*Z_keep + Z_adjust + (Z_value == 0)*(Z_adjust_source != 0);
        state->C = (state->C)*C_keep + C_adjust + (C_direct_value)*(C_adjust_direct != 0);
    }
};

struct Region_Write {
    flag_t address_write_OK = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        int16u_t address = (NULL_ADDRESS_WRITE)*(1-address_write_OK) + (computation_state->address)*address_write_OK;
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
    const Region_BooleanLogic boolean_logic;
    const Region_JSR_RTS_RTI jsr_rts_rti;
    const Region_Rewire rewire;
    const Region_Branch branch;
    const Region_Flags flags;
    const Region_Write write;
    const Region_ProgramCounter program_counter;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        wire.transition(state, computation_state);
        boolean_logic.transition(state, computation_state);
        jsr_rts_rti.transition(state, computation_state);
        rewire.transition(state, computation_state);
        branch.transition(state, computation_state);
        flags.transition(state, computation_state);
        write.transition(state, computation_state);
        program_counter.transition(state, computation_state);
    }
};
