import numpy as np
import behaviors

def byte(data):
    return data & 0x00FF

#-

STACK_ZERO = 0x0100

#-

class ComputationState:
    NULL_ADDRESS = -1
    def __init__(self, address=NULL_ADDRESS, value1=0, value2=0, value3=0):
        self.value = value1
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
        result = byte((computation_state.value | state.A)*self.OR_A +
                      (computation_state.value ^ state.A)*self.XOR_A +
                      (computation_state.value & state.A)*self.AND_A)

        state.A = state.A*(1-self.A_wire) + result*self.A_wire
        computation_state.value1 = computation_state.value1*(1-self.value1_wire) + result*self.value1_wire
        computation_state.value2 = computation_state.value2*(1-self.value2_wire) + result*self.value2_wire
        computation_state.value3 = computation_state.value3*(1-self.value3_wire) + result*self.value3_wire


class Region_ProgramCounter:
    def __init__(self, PC_keep=1, PC_address_adjust=0):
        self.PC_keep = PC_keep
        self.PC_address_adjust = PC_address_adjust

    def transition(self, state, computation_state):
        state.program_counter = state.program_counter*self.PC_keep + (computation_state.address)*self.PC_address_adjust

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
        special_status_bits = behaviors.Behaviors.read_special_status_bits_on_pull(state, computation_state.value1)
        computation_state.value1 = computation_state.value1*(1-self.read_special_status_bits) + (special_status_bits)*self.read_special_status_bits

        if 0 and self.value1_from_read:
            print(f'[----READ:[{STACK_ZERO + state.stack_offset:04X}]={computation_state.value1:02X}----]')

class Region_StackWrite:
    def __init__(self, write_from_value1=0):
        self.write_from_value1 = write_from_value1

    def transition(self, state, computation_state):
        address = ComputationState.NULL_ADDRESS*(1-self.write_from_value1) + (STACK_ZERO + state.stack_offset)*self.write_from_value1
        state.memory[address] = computation_state.value1
        if 0 and self.write_from_value1:
            print(f'[----WRITE:[{address:04X}]={computation_state.value1:02X}----]')

class Region_Rewire:
    def __init__(self, value1_keep=1, value1_from_A=0, value1_from_X=0, value1_from_Y=0,
                       value1_from_P_push_bits=0, value1_from_stack_offset=0,
                       value2_keep=1, value2_from_bit6=0,
                       A_from_value1=0, A_from_X=0, A_from_Y=0,
                       X_from_A=0, X_from_stack_offset=0,
                       Y_from_A=0,
                       stack_offset_from_X=0
                       ):
        # self.value1_keep = value1_keep
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

        self.X_from_A = X_from_A
        self.X_from_stack_offset = X_from_stack_offset

        self.Y_from_A = Y_from_A

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

        any_X = self.X_from_A + self.X_from_stack_offset
        new_X = state.X*(1-any_X) + ((state.A)*self.X_from_A +
                                             (state.stack_offset)*self.X_from_stack_offset)

        any_Y = self.Y_from_A
        new_Y = state.Y*(1-any_Y) + (state.A)*self.Y_from_A

        new_stack_offset = state.stack_offset*(1-self.stack_offset_from_X) + (state.X)*self.stack_offset_from_X

        computation_state.value1 = new_value1
        computation_state.value2 = new_value2
        state.A = new_A
        state.X = new_X
        state.Y = new_Y
        state.stack_offset = new_stack_offset

class Region_Write:
    def __init__(self, address_write=0):
        self.address_write = address_write

    def transition(self, state, computation_state):
        address = ComputationState.NULL_ADDRESS*(1-self.address_write) + computation_state.address*self.address_write
        state.memory[address] = computation_state.value1

class Region_AXY:
    def __init__(self, A_keep=1, A_adjust=0, A_value_adjust=0,
                       X_keep=1, X_adjust=0, X_value_adjust=0,
                       Y_keep=1, Y_adjust=0, Y_value_adjust=0):
        self.A_keep, self.A_adjust, self.A_value_adjust = A_keep, A_adjust, A_value_adjust
        self.X_keep, self.X_adjust, self.X_value_adjust = X_keep, X_adjust, X_value_adjust
        self.Y_keep, self.Y_adjust, self.Y_value_adjust = Y_keep, Y_adjust, Y_value_adjust

    def transition(self, state, computation_state):
        state.A = byte(state.A*self.A_keep + self.A_adjust + (computation_state.value)*self.A_value_adjust)
        state.X = byte(state.X*self.X_keep + self.X_adjust + (computation_state.value)*self.X_value_adjust)
        state.Y = byte(state.Y*self.Y_keep + self.Y_adjust + (computation_state.value)*self.Y_value_adjust)

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

class RegionComposition:
    def __init__(self, program_counter=Region_ProgramCounter(),
                       axy=Region_AXY(),
                       compare=Region_Compare(),
                       stack_offset1=Region_StackOffset(),
                       stack_read=Region_StackRead(),
                       boolean_logic=Region_BooleanLogic(),
                       rewire=Region_Rewire(),
                       branch=Region_Branch(),
                       stack_write=Region_StackWrite(),
                       stack_offset2=Region_StackOffset(),
                       flags=Region_Flags(),
                       flags_byte=Region_FlagsByte(),
                       write=Region_Write()):

        regions = [
            (program_counter, Region_ProgramCounter),
            (axy, Region_AXY),
            (compare, Region_Compare),
            (stack_offset1, Region_StackOffset),
            (stack_read, Region_StackRead),
            (boolean_logic, Region_BooleanLogic),
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
