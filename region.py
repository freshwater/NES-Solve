import numpy as np
import behaviors

def byte(data):
    return data & 0x00FF

#-

STACK_ZERO = 0x0100

#-

class ComputationState:
    NULL_ADDRESS = -2
    def __init__(self, data0, data1, data2,
                 address=NULL_ADDRESS,
                 value1=0, value2=0, value3=0):
        self.data0 = data0
        self.data1 = data1
        self.data2 = data2

        self.value1 = value1
        self.value2 = value2
        self.value3 = value3

        self.address = address

    def __str__(self):
        return str((self.value1, self.value2, self.value3))

class Wire:
    NULL = 0
    VALUE1 = 1
    VALUE2 = 2
    VALUE3 = 3

class Region:
    def transition(state, computation_state):
        assert False, False

class Region_Flags:
    def __init__(self, N_keep=1, N_adjust=0, N_adjust_source=Wire.NULL,
                       O_keep=1, O_adjust=0, O_adjust_source=Wire.NULL,
                       U_keep=1, U_adjust=0,
                       B_keep=1, B_adjust=0,
                       D_keep=1, D_adjust=0,
                       I_keep=1, I_adjust=0,
                       Z_keep=1, Z_adjust=0, Z_adjust_source=Wire.NULL,
                       C_keep=1, C_adjust=0, C_adjust_direct=Wire.NULL):

        self.N_keep, self.N_adjust = N_keep, N_adjust
        self.O_keep, self.O_adjust = O_keep, O_adjust
        self.U_keep, self.U_adjust = U_keep, U_adjust
        self.B_keep, self.B_adjust = B_keep, B_adjust
        self.D_keep, self.D_adjust = D_keep, D_adjust
        self.I_keep, self.I_adjust = I_keep, I_adjust
        self.Z_keep, self.Z_adjust = Z_keep, Z_adjust
        self.C_keep, self.C_adjust = C_keep, C_adjust

        self.N_adjust_source = N_adjust_source
        self.O_adjust_source = O_adjust_source
        self.Z_adjust_source = Z_adjust_source
        self.C_adjust_direct = C_adjust_direct

    def transition(self, state, computation_state):
        N_value = ((self.N_adjust_source == 1)*computation_state.value1 +
                   (self.N_adjust_source == 2)*computation_state.value2 +
                   (self.N_adjust_source == 3)*computation_state.value3)

        O_value = ((self.O_adjust_source == 1)*computation_state.value1 +
                   (self.O_adjust_source == 2)*computation_state.value2 +
                   (self.O_adjust_source == 3)*computation_state.value3)

        Z_value = ((self.Z_adjust_source == 1)*computation_state.value1 +
                   (self.Z_adjust_source == 2)*computation_state.value2 +
                   (self.Z_adjust_source == 3)*computation_state.value3)

        C_direct_value = ((self.C_adjust_direct == 1)*computation_state.value1 +
                          (self.C_adjust_direct == 2)*computation_state.value2 +
                          (self.C_adjust_direct == 3)*computation_state.value3)

        state.N = state.N*self.N_keep + self.N_adjust + (N_value >> 7)*(self.N_adjust_source != 0)
        state.O = state.O*self.O_keep + self.O_adjust + O_value
        state.U = state.U*self.U_keep + self.U_adjust
        state.B = state.B*self.B_keep + self.B_adjust
        state.D = state.D*self.D_keep + self.D_adjust
        state.I = state.I*self.I_keep + self.I_adjust
        state.Z = state.Z*self.Z_keep + self.Z_adjust + (Z_value == 0)*(self.Z_adjust_source != 0)
        state.C = state.C*self.C_keep + self.C_adjust + (C_direct_value)*(self.C_adjust_direct != 0)

class Region_FlagsByte:
    def __init__(self, set_from_value1=0):
        self.set_from_value1 = set_from_value1

    def transition(self, state, computation_state):
        byte = state.status_register_byte()
        new_value = byte*(1-self.set_from_value1) + computation_state.value1*self.set_from_value1
        state.status_register_byte_set(new_value)

class Region_BooleanLogic:
    def __init__(self, OR_A=0, XOR_A=0, AND_A=0,
                 A_wire=0, value1_wire=0, value2_wire=0, value3_wire=0):
        self.A_wire = A_wire
        self.OR_A, self.XOR_A, self.AND_A = OR_A, XOR_A, AND_A
        self.value1_wire, self.value2_wire, self.value3_wire = value1_wire, value2_wire, value3_wire

    def transition(self, state, computation_state):
        result = byte((computation_state.value1 | state.A)*self.OR_A +
                      (computation_state.value1 ^ state.A)*self.XOR_A +
                      (computation_state.value1 & state.A)*self.AND_A)

        state.A = state.A*(1-self.A_wire) + result*self.A_wire
        computation_state.value1 = computation_state.value1*(1-self.value1_wire) + result*self.value1_wire
        computation_state.value2 = computation_state.value2*(1-self.value2_wire) + result*self.value2_wire
        computation_state.value3 = computation_state.value3*(1-self.value3_wire) + result*self.value3_wire

class Region_Arithmetic:
    def __init__(self, value1_increment=0):
        self.value1_increment = value1_increment

    def transition(self, state, computation_state):
        computation_state.value1 = byte((computation_state.value1) + self.value1_increment)

class Region_ADC_SBC:
    def __init__(self, value1_from_ADC=0, value1_from_SBC=0, value2_from_overflow=0, value3_from_carry=0):
        self.value1_from_ADC = value1_from_ADC
        self.value1_from_SBC = value1_from_SBC
        self.value2_from_overflow = value2_from_overflow
        self.value3_from_carry = value3_from_carry

    def transition(self, state, computation_state):
        value1 = (computation_state.value1)*(1-self.value1_from_SBC) + (computation_state.value1 ^ 0xFF)*self.value1_from_SBC
        result = state.A + value1 + state.C
        new_carry = result > 0xFF
        result = byte(result)

        v1 =  ~(state.A ^ value1) & (state.A ^ result) & 0x80
        overflow = v1 > 0

        any_ = self.value1_from_ADC + self.value1_from_SBC
        computation_state.value1 = (computation_state.value1)*(1-any_) + (result)*any_
        computation_state.value2 = (computation_state.value2)*(1-self.value2_from_overflow) + (overflow)*self.value2_from_overflow
        computation_state.value3 = (computation_state.value3)*(1-self.value3_from_carry) + (new_carry)*self.value3_from_carry

class Region_JSR_RTS_RTI:
    def __init__(self, jsr_OK=0, rts_OK=0, rti_OK=0):
        self.jsr_OK = jsr_OK
        self.rts_OK = rts_OK
        self.rti_OK = rti_OK

    def transition(self, state, computation_state):
        program_counter = state.program_counter + (-1)*self.jsr_OK
        pc_H = program_counter >> 8
        pc_L = program_counter & 0x00FF

        any_rt = self.rts_OK + self.rti_OK
        pre_offset = self.rts_OK + self.rti_OK
        post_offset = -self.jsr_OK

        state.stack_offset += self.rti_OK
        read_address = (ComputationState.NULL_ADDRESS)*(1-self.rti_OK) + (STACK_ZERO + state.stack_offset)*self.rti_OK
        new_status = (state.memory[read_address])*self.rti_OK
        new_status = behaviors.Behaviors.read_special_status_bits_on_pull(state, new_status, is_PLP_or_RTI=self.rti_OK)
        new_status = (state.status_register_byte())*(1-self.rti_OK) + (new_status)*self.rti_OK
        state.status_register_byte_set(new_status)

        state.stack_offset += pre_offset

        write_address = (ComputationState.NULL_ADDRESS)*(1-self.jsr_OK) + (STACK_ZERO + state.stack_offset)*self.jsr_OK
        read_address = (ComputationState.NULL_ADDRESS)*(1-any_rt) + (STACK_ZERO + state.stack_offset)*any_rt
        state.memory[write_address] = pc_H
        pc_L = (pc_L)*(1-any_rt) + (state.memory[read_address])*any_rt

        state.stack_offset += pre_offset
        state.stack_offset += post_offset

        write_address = (ComputationState.NULL_ADDRESS)*(1-self.jsr_OK) + (STACK_ZERO + state.stack_offset)*self.jsr_OK
        read_address = (ComputationState.NULL_ADDRESS)*(1-any_rt) + (STACK_ZERO + state.stack_offset)*any_rt
        state.memory[write_address] = pc_L
        pc_H = (pc_H)*(1-any_rt) + (state.memory[read_address])*any_rt

        state.stack_offset += post_offset

        any_ = self.jsr_OK + self.rts_OK + self.rti_OK
        new_program_counter = (program_counter)*(1-any_) + (computation_state.address)*self.jsr_OK + ((pc_H << 8) + pc_L)*any_rt
        new_program_counter = new_program_counter + (1)*self.rts_OK

        state.program_counter = (state.program_counter)*(1-any_) + (new_program_counter)*any_

class Region_BitShift:
    def __init__(self, right_shift=0, right_rotate=0,
                 left_shift=0, left_rotate=0,
                 value1_out=0, value3_from_carry=0):
        self.right_shift = right_shift
        self.right_rotate = right_rotate
        self.left_shift = left_shift
        self.left_rotate = left_rotate

        self.value1_out = value1_out
        self.value3_from_carry = value3_from_carry

    def transition(self, state, computation_state):
        any_ = self.right_shift + self.right_rotate + self.left_shift + self.left_rotate
        any_right = self.right_shift + self.right_rotate
        any_left = self.left_shift + self.left_rotate
        any_rotate = self.left_rotate + self.right_rotate

        new_ = byte(computation_state.value1*(1-any_) + ((computation_state.value1 >> 1)*any_right +
                                                         (computation_state.value1 << 1)*any_left))

        new_carry = (computation_state.value1 & 0x01)*any_right + (computation_state.value1 >> 7)*any_left
        new_ = new_*(1-any_rotate) + (((state.C << 7) | new_)*self.right_rotate +
                                      ((state.C << 0) | new_)*self.left_rotate)

        computation_state.value1 = (computation_state.value1)*(1-self.value1_out) + (new_)*self.value1_out
        computation_state.value3 = (computation_state.value3)*(1-self.value3_from_carry) + (new_carry)*self.value3_from_carry

class Region_Branch:
    def __init__(self, flag_match=0,
                 N_flag_branch=0,
                 O_flag_branch=0,
                 Z_flag_branch=0,
                 C_flag_branch=0):
        self.flag_match = flag_match
        self.N_flag_branch = N_flag_branch
        self.O_flag_branch = O_flag_branch
        self.Z_flag_branch = Z_flag_branch
        self.C_flag_branch = C_flag_branch

    def transition(self, state, computation_state):
        condition = ((state.N == self.flag_match)*self.N_flag_branch +
                     (state.O == self.flag_match)*self.O_flag_branch +
                     (state.Z == self.flag_match)*self.Z_flag_branch +
                     (state.C == self.flag_match)*self.C_flag_branch)

        state.program_counter = state.program_counter + np.int8((computation_state.value1)*condition)

class Region_StackOffset:
    def __init__(self, offset_keep=1, offset_adjust=0):
        self.offset_keep = offset_keep
        self.offset_adjust = offset_adjust

    def transition(self, state, computation_state):
        state.stack_offset = state.stack_offset*self.offset_keep + self.offset_adjust
        value = state.memory[STACK_ZERO + state.stack_offset]

class Region_StackRead:
    def __init__(self, value1_from_read=0, read_special_status_bits=0):
        self.value1_from_read = value1_from_read
        self.read_special_status_bits = read_special_status_bits

    def transition(self, state, computation_state):
        computation_state.value1 = computation_state.value1*(1-self.value1_from_read) + (state.memory[STACK_ZERO + state.stack_offset])*self.value1_from_read
        special_status_bits = behaviors.Behaviors.read_special_status_bits_on_pull(state, computation_state.value1, is_PLP_or_RTI=self.read_special_status_bits)
        computation_state.value1 = computation_state.value1*(1-self.read_special_status_bits) + (special_status_bits)*self.read_special_status_bits

class Region_StackWrite:
    def __init__(self, write_from_value1=0):
        self.write_from_value1 = write_from_value1

    def transition(self, state, computation_state):
        address = ComputationState.NULL_ADDRESS*(1-self.write_from_value1) + (STACK_ZERO + state.stack_offset)*self.write_from_value1
        state.memory[address] = computation_state.value1

class Region_Wire:
    def __init__(self, value1_from_A=0, value1_from_X=0, value1_from_Y=0,
                 value1_from_address=0, value1_from_zeropage=0,
                 value1_from_zeropage_dereference=0, value1_from_absolute_dereference=0,
                 value1_from_indirect_x_dereference=0, value1_from_indirect_y_dereference=0,
                 value1_from_zeropage_x=0, value1_from_absolute_y_dereference=0,
                 value1_from_zeropage_y=0, address_from_absolute=0, address_from_zeropage=0,
                 value1_from_absolute_x_dereference=0,
                 address_from_indirect_x=0, address_from_indirect_y=0,
                 address_from_zeropage_x=0, address_from_absolute_y=0,
                 address_from_zeropage_y=0, address_from_absolute_indirect=0,
                 address_from_absolute_x=0):
        self.value1_from_A = value1_from_A
        self.value1_from_X = value1_from_X
        self.value1_from_Y = value1_from_Y
        self.value1_from_address = value1_from_address
        self.value1_from_zeropage = value1_from_zeropage
        self.value1_from_zeropage_dereference = value1_from_zeropage_dereference
        self.value1_from_absolute_dereference = value1_from_absolute_dereference
        self.value1_from_indirect_x_dereference = value1_from_indirect_x_dereference
        self.value1_from_zeropage_x = value1_from_zeropage_x
        self.value1_from_zeropage_y = value1_from_zeropage_y
        self.value1_from_indirect_y_dereference = value1_from_indirect_y_dereference
        self.value1_from_absolute_y_dereference = value1_from_absolute_y_dereference
        self.value1_from_absolute_x_dereference = value1_from_absolute_x_dereference

        self.address_from_absolute = address_from_absolute
        self.address_from_zeropage = address_from_zeropage
        self.address_from_indirect_x = address_from_indirect_x
        self.address_from_indirect_y = address_from_indirect_y
        self.address_from_absolute_y = address_from_absolute_y
        self.address_from_zeropage_x = address_from_zeropage_x
        self.address_from_zeropage_y = address_from_zeropage_y
        self.address_from_absolute_indirect = address_from_absolute_indirect
        self.address_from_absolute_x = address_from_absolute_x

    def read_if(state, address, condition):
        # print(condition, state.memory[(ComputationState.NULL_ADDRESS)*(1-condition) + (address)*condition], state.memory[-1])
        return state.memory[(ComputationState.NULL_ADDRESS)*(1-condition) + (address)*condition]*condition

    def zeropage_x_dereference(state, computation_state, condition):
        # return state.memory[byte(data1 + state.X)]
        address = (ComputationState.NULL_ADDRESS)*(1-condition) + (byte(computation_state.data1 + state.X))*condition
        return state.memory[address]*condition

    def absolute_x_dereference(state, computation_state, condition):
        # return state.memory[data2*0x0100 + data1 + state.X]
        address = (ComputationState.NULL_ADDRESS)*(1-condition) + (computation_state.data2*0x0100 + computation_state.data1 + state.X)*condition
        return state.memory[address]*condition

    def absolute_x_address(state, computation_state, condition):
        # return data2*0x0100 + data1 + state.X
        return (ComputationState.NULL_ADDRESS)*(1-condition) + (computation_state.data2*0x0100 + computation_state.data1 + state.X)*condition

    def zeropage_x_address(state, computation_state, condition):
        # return byte(data1 + state.X)
        return (ComputationState.NULL_ADDRESS)*(1-condition) + (byte(computation_state.data1 + state.X))*condition

    def indirect_x_dereference(state, computation_state, condition):
        # L = state.memory[byte(data1 + state.X)]
        # H = state.memory[byte(data1 + state.X + 1)]
        # address = H*0x0100 + L
        # return state.memory[address]
        address_L = (ComputationState.NULL_ADDRESS)*(1-condition) + byte(computation_state.data1 + state.X)*condition
        address_H = (ComputationState.NULL_ADDRESS)*(1-condition) + byte(computation_state.data1 + state.X + 1)*condition
        L = state.memory[address_L]
        H = state.memory[address_H]
        address = (ComputationState.NULL_ADDRESS)*(1-condition) + (H*0x0100 + L)*condition
        return state.memory[address]*condition

    def indirect_x_address(state, computation_state, condition):
        # L = state.memory[byte(data1 + state.X)]
        # H = state.memory[byte(data1 + state.X + 1)]
        # return H*0x0100 + L
        address_L = (ComputationState.NULL_ADDRESS)*(1-condition) + byte(computation_state.data1 + state.X)*condition
        address_H = (ComputationState.NULL_ADDRESS)*(1-condition) + byte(computation_state.data1 + state.X + 1)*condition
        L = state.memory[address_L]
        H = state.memory[address_H]
        return (ComputationState.NULL_ADDRESS)*(1-condition) + (H*0x0100 + L)*condition

    def zeropage_y_dereference(state, computation_state, condition):
        # return state.memory[byte(data1 + state.Y)]
        address = (ComputationState.NULL_ADDRESS)*(1-condition) + (byte(computation_state.data1 + state.Y))*condition
        return state.memory[address]*condition

    def zeropage_y_address(state, computation_state, condition):
        # return byte(data1 + state.Y)
        return (ComputationState.NULL_ADDRESS)*(1-condition) + (byte(computation_state.data1 + state.Y))*condition

    def indirect_y_dereference(state, computation_state, condition):
        # address = state.memory[byte(data1+1)]*0x0100 + state.memory[data1]
        # return state.memory[address + state.Y]
        address_L = (ComputationState.NULL_ADDRESS)*(1-condition) + (computation_state.data1)*condition
        address_H = (ComputationState.NULL_ADDRESS)*(1-condition) + (byte(computation_state.data1 + 1))*condition
        address = (ComputationState.NULL_ADDRESS)*(1-condition) + (state.memory[address_H]*0x0100 + state.memory[address_L] + state.Y)*condition
        return state.memory[address]*condition

    def indirect_y_address(state, computation_state, condition):
        # address = state.memory[data1+1]*0x0100 + state.memory[data1]
        # return address + state.Y
        address_L = (ComputationState.NULL_ADDRESS)*(1-condition) + (computation_state.data1)*condition
        address_H = (ComputationState.NULL_ADDRESS)*(1-condition) + (byte(computation_state.data1 + 1))*condition
        address = (ComputationState.NULL_ADDRESS)*(1-condition) + (state.memory[address_H]*0x0100 + state.memory[address_L] + state.Y)*condition
        return address

    def absolute_y_dereference(state, computation_state, condition):
        # address = data2*0x0100 + data1
        # return state.memory[address + state.Y]
        address = (ComputationState.NULL_ADDRESS)*(1-condition) + (computation_state.data2*0x0100 + computation_state.data1 + state.Y)*condition
        return state.memory[address]*condition

    def absolute_y_address(state, computation_state, condition):
        # address = data2*0x0100 + data1 + state.Y
        return (ComputationState.NULL_ADDRESS)*(1-condition) + (computation_state.data2*0x0100 + computation_state.data1 + state.Y)*condition

    def absolute_address_indirect(state, computation_state, condition):
        # L = state.memory[data2*0x0100 + data1]
        # H = state.memory[data2*0x0100 + byte(data1 + 1)]
        # return H*0x0100 + L
        address_L = (ComputationState.NULL_ADDRESS)*(1-condition) + (computation_state.data2*0x0100 + computation_state.data1)*condition
        address_H = (ComputationState.NULL_ADDRESS)*(1-condition) + (computation_state.data2*0x0100 + byte(computation_state.data1 + 1))*condition
        L = state.memory[address_L]
        H = state.memory[address_H]
        return (ComputationState.NULL_ADDRESS)*(1-condition) + (H*0x0100 + L)*condition

    def transition(self, state, computation_state):
        computation_state.value1 = ((state.A)*self.value1_from_A +
                                    (state.X)*self.value1_from_X +
                                    (state.Y)*self.value1_from_Y +
                                    (computation_state.data1)*self.value1_from_zeropage +
                                    Region_Wire.read_if(state, computation_state.data1, self.value1_from_zeropage_dereference) +
                                    Region_Wire.read_if(state, computation_state.data2*0x0100 + computation_state.data1, self.value1_from_absolute_dereference) +
                                    Region_Wire.indirect_x_dereference(state, computation_state, self.value1_from_indirect_x_dereference) +
                                    Region_Wire.zeropage_x_dereference(state, computation_state, self.value1_from_zeropage_x) +
                                    Region_Wire.zeropage_y_dereference(state, computation_state, self.value1_from_zeropage_y) +
                                    Region_Wire.indirect_y_dereference(state, computation_state, self.value1_from_indirect_y_dereference) +
                                    Region_Wire.absolute_y_dereference(state, computation_state, self.value1_from_absolute_y_dereference) +
                                    Region_Wire.absolute_x_dereference(state, computation_state, self.value1_from_absolute_x_dereference))

        computation_state.address = ((computation_state.data2*0x0100 + computation_state.data1)*self.address_from_absolute +
                                     (computation_state.data1)*self.address_from_zeropage +
                                     (Region_Wire.indirect_x_address(state, computation_state, self.address_from_indirect_x))*self.address_from_indirect_x +
                                     (Region_Wire.indirect_y_address(state, computation_state, self.address_from_indirect_y))*self.address_from_indirect_y +
                                     (Region_Wire.absolute_y_address(state, computation_state, self.address_from_absolute_y))*self.address_from_absolute_y +
                                     (Region_Wire.zeropage_x_address(state, computation_state, self.address_from_zeropage_x))*self.address_from_zeropage_x +
                                     (Region_Wire.zeropage_y_address(state, computation_state, self.address_from_zeropage_y))*self.address_from_zeropage_y +
                                     (Region_Wire.absolute_address_indirect(state, computation_state, self.address_from_absolute_indirect))*self.address_from_absolute_indirect +
                                     (Region_Wire.absolute_x_address(state, computation_state, self.address_from_absolute_x))*self.address_from_absolute_x)

    def __str__(self):
        return ' '.join(f'''Region_Wire{{.value1_from_data1={self.value1_from_zeropage},
                                         .value1_from_X={self.value1_from_X},
                                         .address_from_absolute={self.address_from_absolute}}}'''.split())

class Region_Rewire:
    def __init__(self, value1_keep=1, value1_from_A=0, value1_from_X=0, value1_from_Y=0,
                       value1_from_P_push_bits=0, value1_from_stack_offset=0,
                       value2_keep=1, value2_from_bit6=0,
                       A_from_value1=0, A_from_X=0, A_from_Y=0,
                       X_from_value1=0, X_from_A=0, X_from_stack_offset=0,
                       Y_from_value1=0, Y_from_A=0,
                       program_counter_from_address=0,
                       stack_offset_from_X=0):
        self.value1_keep = value1_keep
        self.value1_from_A = value1_from_A
        self.value1_from_X = value1_from_X
        self.value1_from_Y = value1_from_Y
        self.value1_from_P_push_bits = value1_from_P_push_bits
        self.value1_from_stack_offset = value1_from_stack_offset

        self.value2_keep = value2_keep
        self.value2_from_bit6 = value2_from_bit6

        self.A_from_value1 = A_from_value1
        self.A_from_X = A_from_X
        self.A_from_Y = A_from_Y

        self.X_from_value1 = X_from_value1
        self.X_from_A = X_from_A
        self.X_from_stack_offset = X_from_stack_offset

        self.Y_from_value1 = Y_from_value1
        self.Y_from_A = Y_from_A

        self.program_counter_from_address = program_counter_from_address
        self.stack_offset_from_X = stack_offset_from_X

    def transition(self, state, computation_state):
        special_status_bits = behaviors.Behaviors.write_special_status_bits_on_push(state.status_register_byte(), is_PHP_or_BRK=True)

        new_value1 = computation_state.value1*self.value1_keep + ((state.A)*self.value1_from_A +
                                                                  (state.X)*self.value1_from_X +
                                                                  (state.Y)*self.value1_from_Y +
                                                                  (special_status_bits)*self.value1_from_P_push_bits +
                                                                  (state.stack_offset)*self.value1_from_stack_offset)

        new_value2 = computation_state.value2*self.value2_keep + ((computation_state.value1>>6) & 0x01)*self.value2_from_bit6

        any_A = self.A_from_value1 + self.A_from_Y + self.A_from_X
        new_A = state.A*(1-any_A) + ((computation_state.value1)*self.A_from_value1 +
                                     (state.X)*self.A_from_X + 
                                     (state.Y)*self.A_from_Y)

        any_X = self.X_from_A + self.X_from_stack_offset + self.X_from_value1
        new_X = state.X*(1-any_X) + ((state.A)*self.X_from_A +
                                     (state.stack_offset)*self.X_from_stack_offset +
                                     (computation_state.value1)*self.X_from_value1)

        any_Y = self.Y_from_A + self.Y_from_value1
        new_Y = state.Y*(1-any_Y) + ((state.A)*self.Y_from_A +
                                     (computation_state.value1)*self.Y_from_value1)

        new_program_counter = (state.program_counter)*(1-self.program_counter_from_address) + (computation_state.address)*self.program_counter_from_address
        new_stack_offset = state.stack_offset*(1-self.stack_offset_from_X) + (state.X)*self.stack_offset_from_X

        computation_state.value1 = new_value1
        computation_state.value2 = new_value2
        state.A = new_A
        state.X = new_X
        state.Y = new_Y
        state.program_counter = new_program_counter
        state.stack_offset = new_stack_offset

    def __str__(self):
        return ' '.join(f'''Region_Rewire{{.X_from_value1={self.X_from_value1},
                                  .program_counter_from_address={self.program_counter_from_address}}}'''.split())

class Region_Write:
    def __init__(self, address_write=0):
        self.address_write = address_write

    def transition(self, state, computation_state):
        address = ComputationState.NULL_ADDRESS*(1-self.address_write) + computation_state.address*self.address_write
        state.memory[address] = computation_state.value1

    def __str__(self):
        return ' '.join(f'''Region_Write{{.address_write={self.address_write}}}'''.split())

class Region_Compare:
    def __init__(self, A_compare=0, X_compare=0, Y_compare=0, value1_out=0, value3_from_carry=0):
        self.A_compare = A_compare
        self.X_compare = X_compare
        self.Y_compare = Y_compare

        self.value1_out = value1_out
        self.value3_from_carry = value3_from_carry

    def transition(self, state, computation_state):
        result = state.A*self.A_compare + state.X*self.X_compare + state.Y*self.Y_compare - computation_state.value1
        carry = computation_state.value1 <= (state.A*self.A_compare + state.X*self.X_compare + state.Y*self.Y_compare)

        computation_state.value1 = computation_state.value1*(1-self.value1_out) + result*self.value1_out
        computation_state.value3 = computation_state.value3*(1-self.value3_from_carry) + carry*self.value3_from_carry

class Region_ProgramCounter:
    def __init__(self, PC_increment=0):
        self.PC_increment = PC_increment

    def __str__(self):
        return ' '.join(f'''Region_ProgramCounter{{.PC_increment={self.PC_increment}}}'''.split())


class RegionComposition:
    def __init__(self, compare=Region_Compare(),
                       stack_offset1=Region_StackOffset(),
                       stack_read=Region_StackRead(),
                       boolean_logic=Region_BooleanLogic(),
                       arithmetic=Region_Arithmetic(),
                       adc_sbc=Region_ADC_SBC(),
                       jsr_rts_rti=Region_JSR_RTS_RTI(),
                       bit_shift=Region_BitShift(),
                       rewire=Region_Rewire(),
                       branch=Region_Branch(),
                       stack_write=Region_StackWrite(),
                       stack_offset2=Region_StackOffset(),
                       flags=Region_Flags(),
                       flags_byte=Region_FlagsByte(),
                       write=Region_Write()):

        regions = [
            # (wire, Region_Wire),
            (compare, Region_Compare),
            (stack_offset1, Region_StackOffset),
            (stack_read, Region_StackRead),
            (boolean_logic, Region_BooleanLogic),
            (arithmetic, Region_Arithmetic),
            (adc_sbc, Region_ADC_SBC),
            (jsr_rts_rti, Region_JSR_RTS_RTI),
            (bit_shift, Region_BitShift),
            (rewire, Region_Rewire),
            (branch, Region_Branch),
            (stack_write, Region_StackWrite),
            (stack_offset2, Region_StackOffset),
            (flags, Region_Flags),
            (flags_byte, Region_FlagsByte),
            (write, Region_Write)
        ]

        self.regions = [a for a, _ in regions]
        assert all(isinstance(region, type_) for region, type_ in regions), None

    def transition(self, state, computation_state):
        for region in self.regions:
            region.transition(state, computation_state)

    def struct(self, wire_region, byte_count):
        regions = [
            ("wire", wire_region),
            ("rewire", self.regions[8]),
            ("write", self.regions[14]),
            ("program_counter", Region_ProgramCounter(PC_increment=byte_count))
        ]

        return 'RegionComposition{' + ', '.join(f'.{name}=' + str(region) for name, region in regions) + '}'
