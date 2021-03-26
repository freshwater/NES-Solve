
#define STACK_ZERO 0x0100

struct Region_Wire {
    flag_t value1_from_data1 = 0;
    flag16_t value1_from_zeropage_dereference = 0;
    flag16_t value1_from_absolute_dereference = 0;

    flag16_t value1_from_zeropage_x_dereference = 0;
    flag16_t value1_from_absolute_x_dereference = 0;
    flag16_t value1_from_indirect_x_dereference = 0;

    flag16_t value1_from_zeropage_y_dereference = 0;
    flag16_t value1_from_absolute_y_dereference = 0;
    flag16_t value1_from_indirect_y_dereference = 0;

    flag_t value1_from_stack_offset = 0;

    flag_t value1_from_A = 0;
    flag_t value1_from_X = 0;
    flag_t value1_from_Y = 0;

    flag16_t address_from_absolute = 0;
    flag16_t address_from_absolute_y = 0;
    flag16_t address_from_absolute_dereference = 0;
    flag16_t address_from_zeropage = 0;
    flag16_t address_from_zeropage_x = 0;
    flag16_t address_from_absolute_x = 0;
    flag16_t address_from_indirect_x = 0;
    flag16_t address_from_zeropage_y = 0;
    flag16_t address_from_indirect_y = 0;

    int_t cycle_base_increment = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        flag_t any_indirect_x = value1_from_indirect_x_dereference | address_from_indirect_x;
        flag_t any_indirect_y = value1_from_indirect_y_dereference | address_from_indirect_y;
        flag_t any_indirect = any_indirect_x | any_indirect_y;

        int16u_t address_L = (NULL_ADDRESS_READ+0)&(~any_indirect) | (0xFF & (state->data1 + ((state->X)&any_indirect_x)))&any_indirect;
        int16u_t address_H = (NULL_ADDRESS_READ+1)&(~any_indirect) | (0xFF & (state->data1 + ((state->X)&any_indirect_x) + 1))&any_indirect;

        int16u_t address_absolute = (state->data2 << 8) | state->data1;

        address_L = (address_L)&(~address_from_absolute_dereference) | (address_absolute)&address_from_absolute_dereference;
        address_H = (address_H)&(~address_from_absolute_dereference) | ((state->data2 << 8) | 0x00FF&(state->data1 + 1))&address_from_absolute_dereference;

        int16u_t address_L_indirect = system->memory[address_L];
        int16u_t address_H_indirect = system->memory[address_H];
        int16u_t address_HL_indirect = (address_H_indirect << 8) | address_L_indirect;

        flag16_t any_dereference = value1_from_zeropage_dereference | value1_from_absolute_dereference |
                                   value1_from_zeropage_x_dereference | value1_from_absolute_x_dereference | value1_from_indirect_x_dereference |
                                   value1_from_zeropage_y_dereference | value1_from_absolute_y_dereference | value1_from_indirect_y_dereference;

        int address_dereference = (any_dereference == 0x0000) ? NULL_ADDRESS_READ : (state->data1)&value1_from_zeropage_dereference |
                                                                                (address_absolute)&value1_from_absolute_dereference |
                                                                                (address_HL_indirect)&value1_from_indirect_x_dereference |
                                                                                (address_HL_indirect + state->Y)&value1_from_indirect_y_dereference |
                                                                                (address_absolute + state->X)&value1_from_absolute_x_dereference |
                                                                                (address_absolute + state->Y)&value1_from_absolute_y_dereference |
                                                                                (0x00FF&(state->data1 + state->X))&value1_from_zeropage_x_dereference |
                                                                                (0x00FF&(state->data1 + state->Y))&value1_from_zeropage_y_dereference;

        state->value1 = ((state->data1)&value1_from_data1 |
                        (system->memory.read(address_dereference, state))&any_dereference |
                        (state->stack_offset)&value1_from_stack_offset |
                        (state->A)&value1_from_A |
                        (state->X)&value1_from_X |
                        (state->Y)&value1_from_Y);

        state->address = ((address_absolute)&address_from_absolute |
                         (address_HL_indirect)&address_from_absolute_dereference |
                         (state->data1)&address_from_zeropage |
                         (0xFF&(state->data1 + state->X))&address_from_zeropage_x |
                         (address_HL_indirect)&address_from_indirect_x |
                         (address_HL_indirect + state->Y)&address_from_indirect_y |
                         (address_absolute + state->X)&address_from_absolute_x |
                         (address_absolute + state->Y)&address_from_absolute_y |
                         (0x00FF&(state->data1 + state->Y))&address_from_zeropage_y);

        int_t extra_cycle = ((0xFF00&(address_absolute    + state->X)) != (0xFF00&(   address_absolute)))&value1_from_absolute_x_dereference |
                            ((0xFF00&(address_HL_indirect + state->Y)) != (0xFF00&(address_HL_indirect)))&value1_from_indirect_y_dereference |
                            ((0xFF00&(address_absolute    + state->Y)) != (0xFF00&(   address_absolute)))&value1_from_absolute_y_dereference;

        state->instruction_countdown += cycle_base_increment + extra_cycle;
    }
};

struct Region_Compare {
    flag_t A_compare_with_value1 = 0;
    flag_t X_compare_with_value1 = 0;
    flag_t Y_compare_with_value1 = 0;
    flag_t value1_out = 0;
    flag_t value3_from_carry = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        int8u_t lhs = (state->A)&A_compare_with_value1 | (state->X)&X_compare_with_value1 | (state->Y)&Y_compare_with_value1;
        int8u_t result = lhs - state->value1;
        int8u_t carry = (state->value1) <= lhs;

        state->value1 = (state->value1)&(~value1_out) | (result)&value1_out;
        state->value3 = (state->value3)&(~value3_from_carry) | (carry)&value3_from_carry;
    }
};

struct Behaviors {
    __device__
    static int8u_t special_status_bits_on_push(int8u_t status_register, flag_t is_PHP_or_BRK) {
        return status_register | 0x20 | 0x10&is_PHP_or_BRK;
    }

    __device__
    static int8u_t special_status_bits_on_pull(ComputationState* state, int8u_t data, flag_t is_PLP_or_RTI) {
        int8u_t current = state->statusRegisterByteGet();
        int8u_t bits = current & 0x30;
        data &= (0xFF - 0x30);
        data |= bits;

        return (current)&(~is_PLP_or_RTI) | (data)&is_PLP_or_RTI;
    }
};

struct Region_StackRead {
    flag16_t value1_from_stack_read = 0;
    flag_t read_special_status_bits = 0;
    int_signed_t stack_offset_pre_adjust = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        state->stack_offset += stack_offset_pre_adjust;
        // int16u_t address = (NULL_ADDRESS_READ)&(~value1_from_stack_read) | (STACK_ZERO | state->stack_offset)&value1_from_stack_read;
        int address = (value1_from_stack_read == 0x0000) ? NULL_ADDRESS_READ : (STACK_ZERO | state->stack_offset);

        state->value1 = (state->value1)&(~value1_from_stack_read) | (system->memory[address])&value1_from_stack_read;
        int8u_t special_status_bits = Behaviors::special_status_bits_on_pull(state, state->value1, read_special_status_bits);
        state->value1 = (state->value1)&(~read_special_status_bits) | (special_status_bits)&read_special_status_bits;
    }
};

struct Region_BooleanLogic {
    flag_t A_AND_value1 = 0;
    flag_t A_OR_value1 = 0;
    flag_t A_XOR_value1 = 0;
    flag_t value1_out = 0;
    flag_t value3_out = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        int8u_t result = ((state->A & state->value1)&A_AND_value1 |
                          (state->A | state->value1)&A_OR_value1 |
                          (state->A ^ state->value1)&A_XOR_value1);

        state->value1 = (state->value1)&(~value1_out) | (result)&value1_out;
        state->value3 = (state->value3)&(~value3_out) | (result)&value3_out;
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
    void transition(SystemState* system, ComputationState* state) const {
        flag_t any_ = left_shift_from_value1 | right_shift_from_value1 | left_rotate_from_value1 | right_rotate_from_value1;
        flag_t any_right = right_shift_from_value1 | right_rotate_from_value1;
        flag_t any_left = left_shift_from_value1 | left_rotate_from_value1;
        flag_t any_rotate = left_rotate_from_value1 | right_rotate_from_value1;

        int8u_t new_ = (state->value1)&(~any_) | ((state->value1 >> 1)&any_right |
                                                  (state->value1 << 1)&any_left);

        int8u_t new_carry = (state->value1 & 0x01)&any_right | (state->value1 >> 7)&any_left;
        new_ = new_&(~any_rotate) | (((state->C << 7) | new_)&right_rotate_from_value1 |
                                     ((state->C << 0) | new_)&left_rotate_from_value1);

        state->value1 = (state->value1)&(~value1_out) | (new_)&value1_out;
        state->value3 = (state->value3)&(~value3_from_carry) | (new_carry)&value3_from_carry;
    }
};

struct Region_Arithmetic {
    int_signed_t value1_increment = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        state->value1 += value1_increment;
    }
};

struct Region_ADC_SBC {
    flag_t value1_from_ADC = 0;
    flag_t value1_from_SBC = 0;
    flag_t value2_from_overflow = 0;
    flag_t value3_from_carry = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        int8u_t value1 = (state->value1)&(~value1_from_SBC) | (state->value1 ^ 0xFF)&value1_from_SBC;
        int16u_t result = state->A + value1 + state->C;
        int8u_t new_carry = result > 0xFF;
        result &= 0xFF;

        int8u_t overflow =  (~(state->A ^ value1) & (state->A ^ result) & 0x80) > 0;

        flag_t any_ = value1_from_ADC | value1_from_SBC;
        state->value1 = (state->value1)&(~any_) | (result)&any_;
        state->value2 = (state->value2)&(~value2_from_overflow) | (overflow)&value2_from_overflow;
        state->value3 = (state->value3)&(~value3_from_carry) | (new_carry)&value3_from_carry;
    }
};

struct Region_JSR_RTS_RTI {
    flag16_t jsr_OK = 0;
    flag16_t rts_OK = 0;
    flag16_t rti_OK = 0;
    flag16_t brk_OK = 0;
    flag16_t nmi_OK = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        int16u_t program_counter = state->program_counter + ((-1 + 3)&jsr_OK);
        int8u_t pc_H = program_counter >> 8;
        int8u_t pc_L = program_counter & 0x00FF;

        flag16_t any_jump = jsr_OK | brk_OK | nmi_OK;
        flag16_t any_return = rts_OK | rti_OK;

        int_t pre_offset = (1)&any_return;
        int_signed_t post_offset = (-1)&any_jump;

        state->stack_offset += pre_offset;

        // int16u_t write_address = (NULL_ADDRESS_WRITE)&(~any_jump) | (STACK_ZERO | state->stack_offset)&any_jump;
        int write_address = (any_jump == 0x0000) ? NULL_ADDRESS_WRITE : (STACK_ZERO | state->stack_offset);
        int read_address = (any_return == 0x0000) ? NULL_ADDRESS_READ : (STACK_ZERO | state->stack_offset);
        /*write*/system->memory[write_address] = pc_H;
         /*read*/pc_L = (pc_L)&(~any_return) | (system->memory[read_address])&any_return;

        state->stack_offset += pre_offset;
        state->stack_offset += post_offset;

        // write_address = (NULL_ADDRESS_WRITE)&(~any_jump) | (STACK_ZERO | state->stack_offset)&any_jump;
        write_address = (any_jump == 0x0000) ? NULL_ADDRESS_WRITE : (STACK_ZERO | state->stack_offset);
        read_address = (any_return == 0x0000) ? NULL_ADDRESS_READ : (STACK_ZERO | state->stack_offset);
        /*write*/system->memory[write_address] = pc_L;
         /*read*/pc_H = (pc_H)&(~any_return) | (system->memory[read_address])&any_return;

        state->stack_offset += post_offset;

        flag16_t any_ = jsr_OK | rts_OK | rti_OK | brk_OK | nmi_OK;
        int16u_t new_program_counter = (program_counter)&(~any_) | (state->address)&any_jump | ((pc_H << 8) | pc_L)&any_return;
        new_program_counter = new_program_counter + ((1)&rts_OK) + ((-1)&rti_OK);

        state->program_counter = (state->program_counter)&(~any_) | (new_program_counter)&any_;
    }
};

struct Region_Rewire {
    flag_t value1_from_status_push_bits = 0;
    flag_t value2_from_value1_bit6 = 0;

    flag_t A_from_value1 = 0;
    flag_t X_from_value1 = 0;
    flag_t Y_from_value1 = 0;

    flag16_t program_counter_from_address = 0;
    flag_t stack_offset_from_X = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        int8u_t special_status_bits = Behaviors::special_status_bits_on_push(state->statusRegisterByteGet(), value1_from_status_push_bits);

        state->A = (state->A)&(~A_from_value1) | (state->value1)&A_from_value1;
        state->X = (state->X)&(~X_from_value1) | (state->value1)&X_from_value1;
        state->Y = (state->Y)&(~Y_from_value1) | (state->value1)&Y_from_value1;

        state->value1 = (state->value1)&(~value1_from_status_push_bits) | (special_status_bits)&value1_from_status_push_bits;
        state->value2 = (state->value2)&(~value2_from_value1_bit6) | ((state->value1 >> 6) & 0x01)&value2_from_value1_bit6;
        state->program_counter = (state->program_counter)&(~program_counter_from_address) | (state->address)&program_counter_from_address;
        state->stack_offset = (state->stack_offset)&(~stack_offset_from_X) | (state->X)&stack_offset_from_X;
    }
};

struct Region_Branch {
    bit_t flag_match = 0;
    flag_t N_flag_branch = 0;
    flag_t O_flag_branch = 0;
    flag_t Z_flag_branch = 0;
    flag_t C_flag_branch = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        bool condition = ((state->N == flag_match)&N_flag_branch |
                          (state->O == flag_match)&O_flag_branch |
                          (state->Z == flag_match)&Z_flag_branch |
                          (state->C == flag_match)&C_flag_branch);

        state->program_counter = state->program_counter + ((int8_t)(state->value1))*condition;

        state->instruction_countdown += condition;
    }
};

struct Region_Write {
    flag16_t memory_write_value1 = 0;
    flag16_t oam_memory_write_value1 = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        int address = (memory_write_value1 == 0x0000) ? NULL_ADDRESS_WRITE : state->address;
        address = (oam_memory_write_value1 == 0x0000) ? address : state->address;

        system->memory.write(address, state->value1, oam_memory_write_value1, state);
    }
};

struct Region_StackWrite {
    flag16_t stack_write_value1 = 0;
    int_signed_t stack_offset_post_adjust = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        // int16u_t address = (NULL_ADDRESS_WRITE)&(~stack_write_value1) | (STACK_ZERO | state->stack_offset)&stack_write_value1;
        int address = (stack_write_value1 == 0x0000) ? NULL_ADDRESS_WRITE : (STACK_ZERO | state->stack_offset);
        system->memory[address] = state->value1;

        state->stack_offset += stack_offset_post_adjust;
    }
};

struct Region_ImplementationState {
    flag_t store_write_from_value1 = 0;
    flag_t value1_read_from_store = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        state->store = (state->store)&(~store_write_from_value1) | (state->value1)&store_write_from_value1;
        state->value1 = (state->value1)&(~value1_read_from_store) | (state->store)&value1_read_from_store;
    }
};

struct Region_Flags {
    flag_t N_keep = 0xFF;
    int_t  N_adjust = 0;
    int_t  N_adjust_source = 0;
    flag_t O_keep = 0xFF;
    int_t  O_adjust = 0;
    int_t  O_adjust_direct = 0;
    flag_t D_keep = 0xFF;
    int_t  D_adjust = 0;
    flag_t I_keep = 0xFF;
    int_t  I_adjust = 0;
    flag_t Z_keep = 0xFF;
    int_t  Z_adjust = 0;
    int_t  Z_adjust_source = 0;
    flag_t C_keep = 0xFF;
    int_t  C_adjust = 0;
    int_t  C_adjust_direct = 0;
    flag_t set_byte_from_value1 = 0;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        int8u_t N_value = ((state->value1)*(N_adjust_source == 1) |
                           (state->value2)*(N_adjust_source == 2) |
                           (state->value3)*(N_adjust_source == 3));

        int8u_t O_direct_value = ((state->value1)*(O_adjust_direct == 1) |
                                  (state->value2)*(O_adjust_direct == 2) |
                                  (state->value3)*(O_adjust_direct == 3));

        int8u_t Z_value = ((state->value1)*(Z_adjust_source == 1) |
                           (state->value2)*(Z_adjust_source == 2) |
                           (state->value3)*(Z_adjust_source == 3));

        int8u_t C_direct_value = ((state->value1)*(C_adjust_direct == 1) |
                                  (state->value2)*(C_adjust_direct == 2) |
                                  (state->value3)*(C_adjust_direct == 3));

        state->N = ((state->N)&N_keep) + N_adjust + (N_value >> 7)*(N_adjust_source != 0);
        state->O = ((state->O)&O_keep) + O_adjust + (O_direct_value)*(O_adjust_direct != 0);
        state->D = ((state->D)&D_keep) + D_adjust;
        state->Z = ((state->Z)&Z_keep) + Z_adjust + (Z_value == 0)*(Z_adjust_source != 0);
        state->C = ((state->C)&C_keep) + C_adjust + (C_direct_value)*(C_adjust_direct != 0);
        state->I = ((state->I)&I_keep) + I_adjust;

        state->N = (state->N)&(~set_byte_from_value1) | (state->value1 >> 7)&set_byte_from_value1;
        state->O = (state->O)&(~set_byte_from_value1) | (state->value1 >> 6)&set_byte_from_value1;
        state->U = (state->U)&(~set_byte_from_value1) | (state->value1 >> 5)&set_byte_from_value1;
        state->B = (state->B)&(~set_byte_from_value1) | (state->value1 >> 4)&set_byte_from_value1;
        state->D = (state->D)&(~set_byte_from_value1) | (state->value1 >> 3)&set_byte_from_value1;
        state->I = (state->I)&(~set_byte_from_value1) | (state->value1 >> 2)&set_byte_from_value1;
        state->Z = (state->Z)&(~set_byte_from_value1) | (state->value1 >> 1)&set_byte_from_value1;
        state->C = (state->C)&(~set_byte_from_value1) | (state->value1 >> 0)&set_byte_from_value1;

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
    void transition(SystemState* system, ComputationState* state) const {
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
    const Region_ImplementationState implementation_state;
    const Region_Flags flags;
    const Region_Write write;
    const Region_StackWrite stack_write;
    const Region_ProgramCounter program_counter;

    __device__
    void transition(SystemState* system, ComputationState* state) const {
        wire.transition(system, state);
        compare.transition(system, state);
        boolean_logic.transition(system, state);
        bit_shift.transition(system, state);
        arithmetic.transition(system, state);
        stack_read.transition(system, state);
        adc_sbc.transition(system, state);
        jsr_rts_rti.transition(system, state);
        branch.transition(system, state);
        rewire.transition(system, state);
        implementation_state.transition(system, state);
        flags.transition(system, state);
        write.transition(system, state);
        stack_write.transition(system, state);
        program_counter.transition(system, state);
    }
};