
struct Region_Wire {
    flag_t value1_from_data1 = 0;
    flag_t value1_from_zeropage_dereference = 0;
    flag_t value1_from_absolute_dereference = 0;

    flag_t value1_from_zeropage_x_dereference = 0;
    flag_t value1_from_absolute_x_dereference = 0;
    flag_t value1_from_indirect_x_dereference = 0;

    flag_t value1_from_zeropage_y_dereference = 0;
    flag_t value1_from_absolute_y_dereference = 0;
    flag_t value1_from_indirect_y_dereference = 0;

    flag_t value1_from_stack_offset = 0;

    flag_t value1_from_A = 0;
    flag_t value1_from_X = 0;
    flag_t value1_from_Y = 0;

    flag_t address_from_absolute = 0;
    flag_t address_from_absolute_y = 0;
    flag_t address_from_absolute_dereference = 0;
    flag_t address_from_zeropage = 0;
    flag_t address_from_zeropage_x = 0;
    flag_t address_from_absolute_x = 0;
    flag_t address_from_indirect_x = 0;
    flag_t address_from_zeropage_y = 0;
    flag_t address_from_indirect_y = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        flag_t any_indirect = value1_from_indirect_x_dereference + value1_from_indirect_y_dereference + address_from_indirect_x + address_from_indirect_y;
        flag_t any_indirect_x = value1_from_indirect_x_dereference + address_from_indirect_x;

        int16u_t address_L = (NULL_ADDRESS_READ+0)*(1-any_indirect) + (0xFF & (computation_state->data1 + (state->X)*any_indirect_x))*any_indirect;
        int16u_t address_H = (NULL_ADDRESS_READ+1)*(1-any_indirect) + (0xFF & (computation_state->data1 + (state->X)*any_indirect_x + 1))*any_indirect;

        address_L = (address_L)*(1-address_from_absolute_dereference) + ((computation_state->data2 << 8) + computation_state->data1)*address_from_absolute_dereference;
        address_H = (address_H)*(1-address_from_absolute_dereference) + ((computation_state->data2 << 8) + (0xFF & (computation_state->data1 + 1)))*address_from_absolute_dereference;

        int16u_t address_zeropage_dereference = (NULL_ADDRESS_READ)*(1-value1_from_zeropage_x_dereference) + (0xFF & (computation_state->data1 + state->X))*value1_from_zeropage_x_dereference;
        int16u_t address_zeropage_y_dereference = (NULL_ADDRESS_READ)*(1-value1_from_zeropage_y_dereference) + (0xFF & (computation_state->data1 + state->Y))*value1_from_zeropage_y_dereference;

        int16u_t address_L_indirect = state->memory[address_L];
        int16u_t address_H_indirect = state->memory[address_H];

        flag_t any_dereference = value1_from_zeropage_dereference + value1_from_absolute_dereference + value1_from_absolute_y_dereference +
                                 value1_from_indirect_x_dereference + value1_from_indirect_y_dereference + value1_from_zeropage_x_dereference +
                                 value1_from_zeropage_y_dereference + value1_from_absolute_x_dereference;

        int16u_t address_x_indirect = (NULL_ADDRESS_READ)*(1-value1_from_indirect_x_dereference) + ((address_H_indirect << 8) + address_L_indirect)*value1_from_indirect_x_dereference;

        int16u_t address_dereference = (NULL_ADDRESS_READ)*(1-any_dereference) + ((computation_state->data1)*value1_from_zeropage_dereference +
                                                                                  ((computation_state->data2 << 8) + computation_state->data1)*value1_from_absolute_dereference +
                                                                                  ((address_H_indirect << 8) + address_L_indirect)*value1_from_indirect_x_dereference +
                                                                                  ((address_H_indirect << 8) + address_L_indirect + (state->Y)*value1_from_indirect_y_dereference)*value1_from_indirect_y_dereference +
                                                                                  ((computation_state->data2 << 8) + computation_state->data1 + state->X)*value1_from_absolute_x_dereference +
                                                                                  ((computation_state->data2 << 8) + computation_state->data1 + state->Y)*value1_from_absolute_y_dereference +
                                                                                  (address_zeropage_dereference)*value1_from_zeropage_x_dereference +
                                                                                  (address_zeropage_y_dereference)*value1_from_zeropage_y_dereference);

        computation_state->value1 = ((computation_state->data1)*value1_from_data1 +
                                     (state->memory[address_dereference])*any_dereference +
                                     (state->stack_offset)*value1_from_stack_offset +
                                     (state->A)*value1_from_A +
                                     (state->X)*value1_from_X +
                                     (state->Y)*value1_from_Y);

        address_x_indirect = (NULL_ADDRESS_READ)*(1-address_from_indirect_x) + ((address_H_indirect << 8) + address_L_indirect)*address_from_indirect_x;
        int16u_t address_y_indirect = (NULL_ADDRESS_READ)*(1-address_from_indirect_y) + ((address_H_indirect << 8) + address_L_indirect + (state->Y)*address_from_indirect_y)*address_from_indirect_y;

        computation_state->address = (((computation_state->data2 << 8) | computation_state->data1)*address_from_absolute +
                                      (computation_state->data1)*address_from_zeropage +
                                      (0xFF & (computation_state->data1 + state->X))*address_from_zeropage_x +
                                      (address_x_indirect)*address_from_indirect_x +
                                      (address_y_indirect)*address_from_indirect_y +
                                      ((address_H_indirect << 8) + address_L_indirect)*address_from_absolute_dereference +
                                      (0xFF & (computation_state->data1 + state->Y))*address_from_zeropage_y +
                                      ((computation_state->data2 << 8) + computation_state->data1 + state->X)*address_from_absolute_x +
                                      ((computation_state->data2 << 8) + computation_state->data1 + state->Y)*address_from_absolute_y);
    }
};

struct Region_Compare {
    flag_t A_compare_with_value1 = 0;
    flag_t X_compare_with_value1 = 0;
    flag_t Y_compare_with_value1 = 0;
    flag_t value1_out = 0;
    flag_t value3_from_carry = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        int8u_t lhs = (state->A)*A_compare_with_value1 + (state->X)*X_compare_with_value1 + (state->Y)*Y_compare_with_value1;
        int8u_t result = lhs - computation_state->value1;
        int8u_t carry = (computation_state->value1) <= lhs;

        computation_state->value1 = (computation_state->value1)*(1-value1_out) + (result)*value1_out;
        computation_state->value3 = (computation_state->value3)*(1-value3_from_carry) + (carry)*value3_from_carry;
    }
};

struct Behaviors {
    __device__
    static int8u_t special_status_bits_on_push(int8u_t status_register, flag_t is_PHP_or_BRK) {
        return status_register | 0x20 | (is_PHP_or_BRK ? 0x10 : 0x00);
    }

    __device__
    static int8u_t special_status_bits_on_pull(SystemState* state, int8u_t data, flag_t is_PLP_or_RTI) {
        int8u_t current = state->statusRegisterByteGet();
        int8u_t bits = current & 0x30;
        data &= (0xFF - 0x30);
        data |= bits;

        return (current)*(1-is_PLP_or_RTI) + (data)*is_PLP_or_RTI;
    }
};

struct Region_StackRead {
    flag_t value1_from_stack_read = 0;
    flag_t read_special_status_bits = 0;
    int_signed_t stack_offset_pre_adjust = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        state->stack_offset += stack_offset_pre_adjust;
        int16u_t address = (NULL_ADDRESS_READ)*(1-value1_from_stack_read) + (STACK_ZERO + state->stack_offset)*value1_from_stack_read;

        computation_state->value1 = (computation_state->value1)*(1-value1_from_stack_read) + (state->memory[address])*value1_from_stack_read;
        int8u_t special_status_bits = Behaviors::special_status_bits_on_pull(state, computation_state->value1, read_special_status_bits);
        computation_state->value1 = (computation_state->value1)*(1-read_special_status_bits) + (special_status_bits)*read_special_status_bits;
    }
};

struct Region_BooleanLogic {
    flag_t A_AND_value1 = 0;
    flag_t A_OR_value1 = 0;
    flag_t A_XOR_value1 = 0;
    flag_t value1_out = 0;
    flag_t value3_out = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        int8u_t result = ((state->A & computation_state->value1)*A_AND_value1 +
                          (state->A | computation_state->value1)*A_OR_value1 +
                          (state->A ^ computation_state->value1)*A_XOR_value1);

        computation_state->value1 = (computation_state->value1)*(1-value1_out) + (result)*value1_out;
        computation_state->value3 = (computation_state->value3)*(1-value3_out) + (result)*value3_out;
    }
};

struct Region_BitShift {
    flag_t left_shift_from_value1 = 0;
    flag_t right_shift_from_value1 = 0;
    flag_t left_rotate_from_value1 = 0;
    flag_t right_rotate_from_value1 = 0;
    flag_t value1_out = 0;
    flag_t value3_from_carry = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        flag_t any_ = left_shift_from_value1 + right_shift_from_value1 + left_rotate_from_value1 + right_rotate_from_value1;
        flag_t any_right = right_shift_from_value1 + right_rotate_from_value1;
        flag_t any_left = left_shift_from_value1 + left_rotate_from_value1;
        flag_t any_rotate = left_rotate_from_value1 + right_rotate_from_value1;

        int8u_t new_ = (computation_state->value1)*(1-any_) + ((computation_state->value1 >> 1)*any_right +
                                                               (computation_state->value1 << 1)*any_left);

        int8u_t new_carry = (computation_state->value1 & 0x01)*any_right + (computation_state->value1 >> 7)*any_left;
        new_ = new_*(1-any_rotate) + (((state->C << 7) | new_)*right_rotate_from_value1 +
                                      ((state->C << 0) | new_)*left_rotate_from_value1);

        computation_state->value1 = (computation_state->value1)*(1-value1_out) + (new_)*value1_out;
        computation_state->value3 = (computation_state->value3)*(1-value3_from_carry) + (new_carry)*value3_from_carry;
    }
};

struct Region_Arithmetic {
    int_signed_t value1_increment = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        computation_state->value1 = 0xFF & (computation_state->value1 + value1_increment);
    }
};

struct Region_ADC_SBC {
    flag_t value1_from_ADC = 0;
    flag_t value1_from_SBC = 0;
    flag_t value2_from_overflow = 0;
    flag_t value3_from_carry = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        int8u_t value1 = (computation_state->value1)*(1-value1_from_SBC) + (computation_state->value1 ^ 0xFF)*value1_from_SBC;
        // int8u_t value1 = computation_state->value1;
        int16u_t result = state->A + value1 + state->C;
        int8u_t new_carry = result > 0xFF;
        result &= 0xFF;

        int8u_t overflow =  (~(state->A ^ value1) & (state->A ^ result) & 0x80) > 0;

        flag_t any_ = value1_from_ADC + value1_from_SBC;
        computation_state->value1 = (computation_state->value1)*(1-any_) + (result)*any_;
        computation_state->value2 = (computation_state->value2)*(1-value2_from_overflow) + (overflow)*value2_from_overflow;
        computation_state->value3 = (computation_state->value3)*(1-value3_from_carry) + (new_carry)*value3_from_carry;
    }
};

struct Region_JSR_RTS_RTI {
    flag_t jsr_OK = 0;
    flag_t rts_OK = 0;
    flag_t rti_OK = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        int16u_t program_counter = state->program_counter + (-1 + 3)*(int)jsr_OK;
        int8u_t pc_H = program_counter >> 8;
        int8u_t pc_L = program_counter & 0x00FF;

        flag_t any_rt = rts_OK + rti_OK;
        int_t pre_offset = rts_OK + rti_OK;
        int_t post_offset = -jsr_OK;

        state->stack_offset += pre_offset;

        int16u_t write_address = (NULL_ADDRESS_WRITE)*(1-jsr_OK) + (STACK_ZERO + state->stack_offset)*jsr_OK;
        int16u_t read_address = (NULL_ADDRESS_READ)*(1-any_rt) + (STACK_ZERO + state->stack_offset)*any_rt;
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
        new_program_counter = new_program_counter + (1)*rts_OK + (-1)*(int_signed_t)rti_OK;

        state->program_counter = (state->program_counter)*(1-any_) + (new_program_counter)*any_;
    }
};

struct Region_Rewire {
    flag_t value1_from_A = 0;
    flag_t value1_from_status_push_bits = 0;
    flag_t value2_from_value1_bit6 = 0;

    flag_t A_from_value1 = 0;
    flag_t X_from_value1 = 0;
    flag_t Y_from_value1 = 0;

    flag_t program_counter_from_address = 0;
    flag_t stack_offset_from_X = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        int8u_t special_status_bits = Behaviors::special_status_bits_on_push(state->statusRegisterByteGet(), value1_from_status_push_bits);

        int8u_t new_A = (state->A)*(1-A_from_value1) + (computation_state->value1)*A_from_value1;
        int8u_t new_X = (state->X)*(1-X_from_value1) + (computation_state->value1)*X_from_value1;
        int8u_t new_Y = (state->Y)*(1-Y_from_value1) + (computation_state->value1)*Y_from_value1;

        flag_t any_value1 = value1_from_A + value1_from_status_push_bits;
        int8u_t new_value1 = (computation_state->value1)*(1-any_value1) + ((state->A)*value1_from_A +
                                                                           (special_status_bits)*value1_from_status_push_bits);
        int8u_t new_value2 = (computation_state->value2)*(1-value2_from_value1_bit6) + ((computation_state->value1 >> 6) & 0x01)*value2_from_value1_bit6;
        int16u_t new_program_counter = (state->program_counter)*(1-program_counter_from_address) + (computation_state->address)*program_counter_from_address;

        int8u_t new_stack_offset = (state->stack_offset)*(1-stack_offset_from_X) + (state->X)*stack_offset_from_X;

        state->A = new_A;
        state->X = new_X;
        state->Y = new_Y;

        computation_state->value1 = new_value1;
        computation_state->value2 = new_value2;

        state->program_counter = new_program_counter;
        state->stack_offset = new_stack_offset;
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

struct Region_Write {
    flag_t memory_write_value1 = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        int16u_t address = (NULL_ADDRESS_WRITE)*(1-memory_write_value1) + (computation_state->address)*memory_write_value1;
        state->memory[address] = computation_state->value1;

    }
};

struct Region_StackWrite {
    flag_t stack_write_value1 = 0;
    int_signed_t stack_offset_post_adjust = 0;

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        int16u_t address = (NULL_ADDRESS_WRITE)*(1-stack_write_value1) + (STACK_ZERO + state->stack_offset)*stack_write_value1;
        state->memory[address] = computation_state->value1;

        state->stack_offset += stack_offset_post_adjust;
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
    flag_t set_byte_from_value1 = 0;

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

        state->N = (state->N)*(1-set_byte_from_value1) + (computation_state->value1 >> 7)*set_byte_from_value1;
        state->O = (state->O)*(1-set_byte_from_value1) + (computation_state->value1 >> 6)*set_byte_from_value1;
        state->U = (state->U)*(1-set_byte_from_value1) + (computation_state->value1 >> 5)*set_byte_from_value1;
        state->B = (state->B)*(1-set_byte_from_value1) + (computation_state->value1 >> 4)*set_byte_from_value1;
        state->D = (state->D)*(1-set_byte_from_value1) + (computation_state->value1 >> 3)*set_byte_from_value1;
        state->I = (state->I)*(1-set_byte_from_value1) + (computation_state->value1 >> 2)*set_byte_from_value1;
        state->Z = (state->Z)*(1-set_byte_from_value1) + (computation_state->value1 >> 1)*set_byte_from_value1;
        state->C = (state->C)*(1-set_byte_from_value1) + (computation_state->value1 >> 0)*set_byte_from_value1;

        state->N &= 0x01;
        state->O &= 0x01;
        state->U &= 0x01;
        state->B &= 0x01;
        state->D &= 0x01;
        state->I &= 0x01;
        state->Z &= 0x01;
        state->C &= 0x01;
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
    const Region_Compare compare;
    const Region_BooleanLogic boolean_logic;
    const Region_BitShift bit_shift;
    const Region_Arithmetic arithmetic;
    const Region_StackRead stack_read;
    const Region_ADC_SBC adc_sbc;
    const Region_JSR_RTS_RTI jsr_rts_rti;
    const Region_Branch branch;
    const Region_Rewire rewire;
    const Region_Flags flags;
    const Region_Write write;
    const Region_StackWrite stack_write;
    const Region_ProgramCounter program_counter;

    __device__
    void dg(SystemState* state, ComputationState* computation_state) const {
        if (threadIdx.x == 7 && state->program_counter >= 0xCDAE) {
            printf("[%X, %X, %X]\n",
                state->program_counter,
                state->memory[0x07FF],
                state->X);
        }
    }

    __device__
    void transition(SystemState* state, ComputationState* computation_state) const {
        wire.transition(state, computation_state);
        compare.transition(state, computation_state);
        boolean_logic.transition(state, computation_state);
        bit_shift.transition(state, computation_state);
        arithmetic.transition(state, computation_state);
        stack_read.transition(state, computation_state);
        adc_sbc.transition(state, computation_state);
        jsr_rts_rti.transition(state, computation_state);
        branch.transition(state, computation_state);
        rewire.transition(state, computation_state);
        flags.transition(state, computation_state);
        write.transition(state, computation_state);
        stack_write.transition(state, computation_state);
        program_counter.transition(state, computation_state);
    }
};
