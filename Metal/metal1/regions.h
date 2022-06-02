//
//  regions.h
//

typedef const uint8_t flag_t;
typedef const uint8_t uint_t;
typedef const int8_t int_t;

#define STACK_ZERO 0x0100

struct Region_Wire {
    flag_t value1_from_data1 : 1;
    flag_t value1_from_stack_offset : 1;

    flag_t value1_from_A : 1;
    flag_t value1_from_X : 1;
    flag_t value1_from_Y : 1;

    flag_t value1_from_zeropage_dereference : 1;
    flag_t value1_from_absolute_dereference : 1;
    flag_t value1_from_zeropage_x_dereference : 1;
    flag_t value1_from_absolute_x_dereference : 1;
    flag_t value1_from_indirect_x_dereference : 1;
    flag_t value1_from_zeropage_y_dereference : 1;
    flag_t value1_from_absolute_y_dereference : 1;
    flag_t value1_from_indirect_y_dereference : 1;

    flag_t address_from_absolute : 1;
    flag_t address_from_absolute_y : 1;
    flag_t address_from_absolute_dereference : 1;
    flag_t address_from_zeropage : 1;
    flag_t address_from_zeropage_x : 1;
    flag_t address_from_absolute_x : 1;
    flag_t address_from_indirect_x : 1;
    flag_t address_from_zeropage_y : 1;
    flag_t address_from_indirect_y : 1;

    uint_t cycle_base_increment;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        int any_indirect_x = value1_from_indirect_x_dereference | address_from_indirect_x;
        int any_indirect_y = value1_from_indirect_y_dereference | address_from_indirect_y;
        int any_indirect = any_indirect_x | any_indirect_y;

        int address_absolute = (state->data2 << 8) | state->data1;
        int address_HL_indirect = 0;

        if (any_indirect | address_from_absolute_dereference) {
            int address_L = any_indirect ? (0xFF & (state->data1 + ((state->X)*any_indirect_x))) : (address_absolute);
            int address_H = any_indirect ? (0xFF & (state->data1 + ((state->X)*any_indirect_x) + 1))
                                         : ((state->data2 << 8) | (0x00FF&(state->data1 + 1)));

            int address_L_indirect = Memory__readMemoryRaw(memory, address_L);
            int address_H_indirect = Memory__readMemoryRaw(memory, address_H);
            address_HL_indirect = (address_H_indirect << 8) | address_L_indirect;
        }

        int any_dereference = value1_from_zeropage_dereference | value1_from_absolute_dereference |
                              value1_from_zeropage_x_dereference | value1_from_absolute_x_dereference | value1_from_indirect_x_dereference |
                              value1_from_zeropage_y_dereference | value1_from_absolute_y_dereference | value1_from_indirect_y_dereference;

        if (any_dereference) {
            int address_dereference = (state->data1)*value1_from_zeropage_dereference +
                                      (address_absolute)*value1_from_absolute_dereference +
                                      (address_HL_indirect)*value1_from_indirect_x_dereference +
                                      (address_HL_indirect + state->Y)*value1_from_indirect_y_dereference +
                                      (address_absolute + state->X)*value1_from_absolute_x_dereference +
                                      (address_absolute + state->Y)*value1_from_absolute_y_dereference +
                                      (0xFF&(state->data1 + state->X))*value1_from_zeropage_x_dereference +
                                      (0x00FF&(state->data1 + state->Y))*value1_from_zeropage_y_dereference;

            state->value1 = Memory__readMemoryLogical(memory, address_dereference);

        } else {

            state->value1 = (state->data1)*value1_from_data1 +
                            (state->stack_offset)*value1_from_stack_offset +
                            (state->A)*value1_from_A +
                            (state->X)*value1_from_X +
                            (state->Y)*value1_from_Y;
        }

        state->address = (address_absolute)*address_from_absolute +
                         (state->data1)*address_from_zeropage +
                         (0xFF&(state->data1 + state->X))*address_from_zeropage_x +
                         (address_HL_indirect)*address_from_absolute_dereference +
                         (address_HL_indirect)*address_from_indirect_x +
                         (address_HL_indirect + state->Y)*address_from_indirect_y +
                         (address_absolute + state->X)*address_from_absolute_x +
                         (address_absolute + state->Y)*address_from_absolute_y +
                         (0x00FF&(state->data1 + state->Y))*address_from_zeropage_y;
    }

    /*
    void transition(SystemState* system, ComputationState* state, Memory& memory) const {
        int extra_cycle = ((0xFF00&(address_absolute    + state->X)) != (0xFF00&(   address_absolute)))*value1_from_absolute_x_dereference +
                          ((0xFF00&(address_HL_indirect + state->Y)) != (0xFF00&(address_HL_indirect)))*value1_from_indirect_y_dereference +
                          ((0xFF00&(address_absolute    + state->Y)) != (0xFF00&(   address_absolute)))*value1_from_absolute_y_dereference;

        state->instruction_countdown += cycle_base_increment + extra_cycle;
    }
    */
};

struct Region_Compare {
    flag_t A_compare_with_value1 : 1;
    flag_t X_compare_with_value1 : 1;
    flag_t Y_compare_with_value1 : 1;

    flag_t value1_out : 1;
    flag_t value3_from_carry : 1;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        int8u_t lhs = (state->A)*A_compare_with_value1 +
                      (state->X)*X_compare_with_value1 +
                      (state->Y)*Y_compare_with_value1;
        int8u_t result = lhs - state->value1;
        int8u_t carry = (state->value1) <= lhs;

        state->value1 = value1_out ? (result) : state->value1;
        state->value3 = value3_from_carry ? (carry) : state->value3;
    }
};

struct Behaviors {
    static int8u_t special_status_bits_on_push(int8u_t status_register, flag_t is_PHP_or_BRK) {
        /*
        https://wiki.nesdev.com/w/index.php/Status_flags#The_B_flag
        Two interrupts (/IRQ and /NMI) and two instructions (PHP and BRK)
        push the flags to the stack. In the byte pushed, bit 5 is always
        set to 1, and bit 4 is 1 if from an instruction (PHP or BRK) or 0
        if from an interrupt line being pulled low (/IRQ or /NMI). This is
        the only time and place where the B flag actually exists: not in
        the status register itself, but in bit 4 of the copy that is
        written to the stack. */
        return status_register | 0x20 | 0x10*is_PHP_or_BRK;
    }

    static int8u_t special_status_bits_on_pull(int8u_t current_status_register, int8u_t data) {
        /*
        https://wiki.nesdev.com/w/index.php/Status_flags#The_B_flag
        Two instructions (PLP and RTI) pull a byte from the stack and set all
        the flags. They ignore bits 5 and 4. */
        return (data & ~0x30) | (current_status_register & 0x30);
    }
};

struct Region_StackRead {
    flag_t value1_from_stack_read : 1;
    flag_t read_special_status_bits : 1;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        state->stack_offset += value1_from_stack_read;

        state->value1 = value1_from_stack_read ? (Memory__readMemoryRaw(memory, STACK_ZERO | state->stack_offset)) : state->value1;
        int8u_t special_status_bits = Behaviors::special_status_bits_on_pull(state->statusRegisterByteGet(), state->value1);
        state->value1 = read_special_status_bits ? (special_status_bits) : state->value1;
    }
};

struct Region_BooleanLogic {
    flag_t A_AND_value1 : 1;
    flag_t A_OR_value1 : 1;
    flag_t A_XOR_value1 : 1;

    flag_t value1_out : 1;
    flag_t value3_out : 1;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        int8u_t result = (state->A & state->value1)*A_AND_value1 +
                         (state->A | state->value1)*A_OR_value1 +
                         (state->A ^ state->value1)*A_XOR_value1;

        state->value1 = value1_out ? (result) : state->value1;
        state->value3 = value3_out ? (result) : state->value3;
    }
};

struct Region_BitShift {
    flag_t left_shift_from_value1 : 1;
    flag_t right_shift_from_value1 : 1;
    flag_t left_rotate_from_value1 : 1;
    flag_t right_rotate_from_value1 : 1;

    flag_t value1_out : 1;
    flag_t value3_from_carry : 1;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        if (left_shift_from_value1 | right_shift_from_value1 | left_rotate_from_value1 | right_rotate_from_value1) {
            flag_t any_left = left_shift_from_value1 | left_rotate_from_value1;
            flag_t any_right = right_shift_from_value1 | right_rotate_from_value1;

            int8u_t new_ = (state->value1 << 1)*any_left + (state->value1 >> 1)*any_right;
            new_ |= (state->C << 0)*left_rotate_from_value1 + (state->C << 7)*right_rotate_from_value1;
            int8u_t new_carry = (state->value1 >> 7)*any_left + (state->value1 & 0x01)*any_right;

            state->value1 = value1_out ? (new_) : state->value1;
            state->value3 = value3_from_carry ? (new_carry) : state->value3;
        }
    }
};

struct Region_Arithmetic {
    int_t value1_increment;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        state->value1 += value1_increment;
    }
};

struct Region_ADC_SBC {
    flag_t value1_from_ADC : 1;
    flag_t value1_from_SBC : 1;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        if (value1_from_ADC | value1_from_SBC) {
            int8u_t value1 = value1_from_SBC ? ~(state->value1) : state->value1;
            int16u_t result = state->A + value1 + state->C;
            int8u_t carry = result > 0xFF;
            result &= 0xFF;

            int8u_t overflow = (~(state->A ^ value1) & (state->A ^ result) & 0x80) > 0;

            state->value1 = result;
            state->value2 = overflow;
            state->value3 = carry;
        }
    }
};

struct Region_JSR_RTS_RTI {
    flag_t jsr_OK : 1;
    flag_t rts_OK : 1;
    flag_t rti_OK : 1;
    flag_t brk_OK : 1;
    flag_t nmi_OK : 1;

    flag_t any_OK : 1;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        if (any_OK) {
            if (jsr_OK) {
                state->program_counter += (-1 + 3)*jsr_OK;

                Memory__writeMemoryRaw(memory, STACK_ZERO | state->stack_offset, state->program_counter >> 8);
                state->stack_offset--;
                Memory__writeMemoryRaw(memory, STACK_ZERO | state->stack_offset, state->program_counter & 0x00FF);
                state->stack_offset--;

                state->program_counter = state->address;
            } else {
                state->stack_offset++;
                uint8_t pc_L = Memory__readMemoryRaw(memory, STACK_ZERO | state->stack_offset);
                state->stack_offset++;
                uint8_t pc_H = Memory__readMemoryRaw(memory, STACK_ZERO | state->stack_offset);

                state->program_counter = ((pc_H << 8) | pc_L) + rts_OK - rti_OK;
            }
        }
    }

    /*
    void transition(SystemState* system, ComputationState* state, Memory& memory) const {
        if (any_OK) {
            if (// jsr_OK |
                brk_OK |
                nmi_OK) {
                // state->program_counter += (-1 + 3)*jsr_OK;

                // memory[STACK_ZERO | state->stack_offset] = state->program_counter >> 8;
                // state->stack_offset--;
                // memory[STACK_ZERO | state->stack_offset] = state->program_counter & 0x00FF;
                // state->stack_offset--;
                // state->program_counter = state->address;
            } else {
                // state->stack_offset++;
                // uint8_t pc_L = memory[STACK_ZERO | state->stack_offset];
                // state->stack_offset++;
                // uint8_t pc_H = memory[STACK_ZERO | state->stack_offset];
                // state->program_counter = ((pc_H << 8) | pc_L) + rts_OK - rti_OK;
            }
        }
    }
    */
};

struct Region_Rewire {
    flag_t value1_from_status_push_bits : 1;
    flag_t value2_from_value1_bit6 : 1;

    flag_t A_from_value1 : 1;
    flag_t X_from_value1 : 1;
    flag_t Y_from_value1 : 1;

    flag_t program_counter_from_address : 1;
    flag_t stack_offset_from_X : 1;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        int8u_t special_status_bits = Behaviors::special_status_bits_on_push(state->statusRegisterByteGet(), value1_from_status_push_bits);

        state->A = A_from_value1 ? (state->value1) : state->A;
        state->X = X_from_value1 ? (state->value1) : state->X;
        state->Y = Y_from_value1 ? (state->value1) : state->Y;

        state->value1 = value1_from_status_push_bits ? (special_status_bits) : state->value1;
        state->value2 = value2_from_value1_bit6 ? ((state->value1 >> 6) & 0x01) : state->value2;

        state->program_counter = program_counter_from_address ? (state->address) : state->program_counter;
        state->stack_offset = stack_offset_from_X ? (state->X) : state->stack_offset;
    }
};

struct Region_Branch {
    flag_t flag_match : 1;

    flag_t N_flag_branch : 1;
    flag_t O_flag_branch : 1;
    flag_t Z_flag_branch : 1;
    flag_t C_flag_branch : 1;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        bool condition = (state->N == flag_match)*N_flag_branch +
                         (state->O == flag_match)*O_flag_branch +
                         (state->Z == flag_match)*Z_flag_branch +
                         (state->C == flag_match)*C_flag_branch;

        state->program_counter = state->program_counter + ((int8_t)(state->value1))*condition;
    }
    /*
    void transition(SystemState* system, ComputationState* state, Memory& memory) const {
        bool condition =  // ((state->N == flag_match)*N_flag_branch +
                          // (state->O == flag_match)*O_flag_branch +
                          // (state->Z == flag_match)*Z_flag_branch +
                          // (state->C == flag_match)*C_flag_branch);

        // state->program_counter = state->program_counter + ((int8_t)(state->value1))*condition;

        state->instruction_countdown += condition;
    }
    */
};

struct Region_Write {
    flag_t memory_write_value1 : 1;
    flag_t oam_memory_write_value1 : 1;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        if (memory_write_value1) {
            Memory__writeMemoryLogical(memory, state->address, state->value1);
        }
    }
    /*
    void transition(SystemState* system, ComputationState* state, Memory& memory) const {
        if ( // memory_write_value1
             | oam_memory_write_value1) {
            memory.write( // state->address, state->value1,
                         oam_memory_write_value1, state);
        }
    }
    */
};

struct Region_StackWrite {
    flag_t stack_write_value1 : 1;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        int address = stack_write_value1 ? (STACK_ZERO | state->stack_offset) : NULL_ADDRESS_WRITE;
        Memory__writeMemoryRaw(memory, address, state->value1);
        state->stack_offset -= stack_write_value1;
    }
};

struct Region_ImplementationState {
    flag_t store_write_from_value1 : 1;
    flag_t value1_read_from_store : 1;

    /*
    void transition(SystemState* system, ComputationState* state, Memory& memory) const {
        state->store = store_write_from_value1 ? (state->value1) : state->store;
        state->value1 = value1_read_from_store ? (state->store) : state->value1;
    }
    */
};

struct Region_Flags {
    flag_t N_keep : 1;
    flag_t N_adjust : 1;
    uint_t N_adjust_source : 2;
    flag_t O_keep : 1;
    flag_t O_adjust : 1;
    uint_t O_adjust_direct : 2;
    flag_t D_keep : 1;
    flag_t D_adjust : 1;
    flag_t I_keep : 1;
    flag_t I_adjust : 1;
    flag_t Z_keep : 1;
    flag_t Z_adjust : 1;
    uint_t Z_adjust_source : 2;
    flag_t C_keep : 1;
    flag_t C_adjust : 1;
    uint_t C_adjust_direct : 2;
    flag_t set_byte_from_value1 : 1;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        int8u_t N_value = (state->value1)*(N_adjust_source == 1);

        int8u_t O_direct_value = (state->value2)*(O_adjust_direct == 2);

        int8u_t Z_value = (state->value1)*(Z_adjust_source == 1) |
                          (state->value3)*(Z_adjust_source == 3);

        int8u_t C_direct_value = (state->value3)*(C_adjust_direct == 3);

        state->N = (state->N)*N_keep + N_adjust + (N_value >> 7);
        state->O = (state->O)*O_keep + O_adjust + (O_direct_value);
        state->D = (state->D)*D_keep + D_adjust;
        // state->I = ((state->I)*I_keep) + I_adjust;
        state->Z = (state->Z)*Z_keep + Z_adjust + (Z_value == 0)*(Z_adjust_source != 0);
        state->C = (state->C)*C_keep + C_adjust + (C_direct_value);

        if (set_byte_from_value1) {
            state->N = (state->value1 >> 7) & 0x01;
            state->O = (state->value1 >> 6) & 0x01;
            state->U = (state->value1 >> 5) & 0x01;
            state->B = (state->value1 >> 4) & 0x01;
            state->D = (state->value1 >> 3) & 0x01;
            // state->I = (state->value1 >> 2) & 0x01;
            state->Z = (state->value1 >> 1) & 0x01;
            state->C = (state->value1 >> 0) & 0x01;
        }
    }

    /*
    void transition(SystemState* system, ComputationState* state, Memory& memory) const {
        int8u_t N_value = // ((state->value1)*(N_adjust_source == 1) |
                           (state->value2)*(N_adjust_source == 2) |
                           (state->value3)*(N_adjust_source == 3));

        int8u_t O_direct_value = ((state->value1)*(O_adjust_direct == 1) |
                                  // (state->value2)*(O_adjust_direct == 2) |
                                  (state->value3)*(O_adjust_direct == 3));

        int8u_t Z_value =  // ((state->value1)*(Z_adjust_source == 1) |
                           (state->value2)*(Z_adjust_source == 2) |
                           // (state->value3)*(Z_adjust_source == 3));

        int8u_t C_direct_value = ((state->value1)*(C_adjust_direct == 1) |
                                  (state->value2)*(C_adjust_direct == 2) |
                                  // (state->value3)*(C_adjust_direct == 3));

        // state->N = ((state->N)*N_keep) + N_adjust + (N_value >> 7);
        // state->O = ((state->O)*O_keep) + O_adjust + (O_direct_value);
        // state->D = ((state->D)*D_keep) + D_adjust;
        // state->Z = ((state->Z)*Z_keep) + Z_adjust + (Z_value == 0)*(Z_adjust_source != 0);
        // state->C = ((state->C)*C_keep) + C_adjust + (C_direct_value);
        state->I = ((state->I)*I_keep) + I_adjust;

        if (set_byte_from_value1) {
            // state->N = (state->value1 >> 7) & 0x01;
            // state->O = (state->value1 >> 6) & 0x01;
            // state->U = (state->value1 >> 5) & 0x01;
            // state->B = (state->value1 >> 4) & 0x01;
            // state->D = (state->value1 >> 3) & 0x01;
            state->I = (state->value1 >> 2) & 0x01;
            // state->Z = (state->value1 >> 1) & 0x01;
            // state->C = (state->value1 >> 0) & 0x01;
        }
    }
    */
};

struct Region_ProgramCounter {
    uint_t PC_increment;

    void transition(thread ComputationState* state, device Memory& memory) const constant {
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

    void transition(thread ComputationState* state, device Memory& memory) const constant {
        // if (state->opcode == NOP_instruction) {
        //     return;
        // }

        wire.transition(state, memory);
        compare.transition(state, memory);
        boolean_logic.transition(state, memory);
        bit_shift.transition(state, memory);
        arithmetic.transition(state, memory);
        stack_read.transition(state, memory);
        adc_sbc.transition(state, memory);
        jsr_rts_rti.transition(state, memory);
        branch.transition(state, memory);
        rewire.transition(state, memory);
        // implementation_state.transition(system, state, memory);
        flags.transition(state, memory);
        write.transition(state, memory);
        stack_write.transition(state, memory);
        program_counter.transition(state, memory);
    }
};

