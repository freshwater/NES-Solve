import numpy as np

def byte(data):
    return data & 0x00FF

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

class Region_Flags:
    def __init__(self, N_keep=1, N_adjust=0, N_adjust_source=Wire.NULL,
                       O_keep=1, O_adjust=0, O_adjust_source=Wire.NULL,
                       U_keep=1, U_adjust=0,
                       B_keep=1, B_adjust=0,
                       D_keep=1, D_adjust=0,
                       I_keep=1, I_adjust=0,
                       Z_keep=1, Z_adjust=0, Z_adjust_source=Wire.NULL,
                       C_keep=1, C_adjust=0):

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

    def transition(self, state, computation_state):
        N_keep, N_adjust = self.N_keep, self.N_adjust
        O_keep, O_adjust = self.O_keep, self.O_adjust
        U_keep, U_adjust = self.U_keep, self.U_adjust
        B_keep, B_adjust = self.B_keep, self.B_adjust
        D_keep, D_adjust = self.D_keep, self.D_adjust
        I_keep, I_adjust = self.I_keep, self.I_adjust
        Z_keep, Z_adjust = self.Z_keep, self.Z_adjust
        C_keep, C_adjust = self.C_keep, self.C_adjust

        N_adjust_source = self.N_adjust_source
        O_adjust_source = self.O_adjust_source
        Z_adjust_source = self.Z_adjust_source

        N_value = ((N_adjust_source == 1)*computation_state.value1 +
                   (N_adjust_source == 2)*computation_state.value2 +
                   (N_adjust_source == 3)*computation_state.value3)

        O_value = ((O_adjust_source == 1)*computation_state.value1 +
                   (O_adjust_source == 2)*computation_state.value2 +
                   (O_adjust_source == 3)*computation_state.value3)
                          
        Z_value = ((Z_adjust_source == 1)*computation_state.value1 +
                   (Z_adjust_source == 2)*computation_state.value2 +
                   (Z_adjust_source == 3)*computation_state.value3)

        state.N = state.N*N_keep + N_adjust + (N_value >> 7)*(N_adjust_source != 0)
        state.O = state.O*O_keep + O_adjust + O_value
        state.U = state.U*U_keep + U_adjust
        state.B = state.B*B_keep + B_adjust
        state.D = state.D*D_keep + D_adjust
        state.I = state.I*I_keep + I_adjust
        state.Z = state.Z*Z_keep + Z_adjust + (Z_value == 0)*(Z_adjust_source != 0)
        state.C = state.C*C_keep + C_adjust

class RegionBooleanLogic:
    def __init__(self, OR_A=0, XOR_A=0, AND_A=0,
                 A_wire=0, value1_wire=0, value2_wire=0, value3_wire=0):
        self.A_wire = A_wire
        self.OR_A, self.XOR_A, self.AND_A = OR_A, XOR_A, AND_A
        self.value1_wire, self.value2_wire, self.value3_wire = value1_wire, value2_wire, value3_wire

    def transition(self, state, computation_state):
        value1_wire, value2_wire, value3_wire = self.value1_wire, self.value2_wire, self.value3_wire

        result = byte((computation_state.value | state.A)*self.OR_A +
                      (computation_state.value ^ state.A)*self.XOR_A +
                      (computation_state.value & state.A)*self.AND_A)

        state.A = state.A*(1-self.A_wire) + result*self.A_wire
        computation_state.value1 = computation_state.value1*(1-value1_wire) + result*self.value1_wire
        computation_state.value2 = computation_state.value2*(1-value2_wire) + result*self.value2_wire
        computation_state.value3 = computation_state.value3*(1-value3_wire) + result*self.value3_wire


class Region1:
    def __init__(self, PC_keep=1, PC_value_adjust=0):
        self.PC_keep = PC_keep
        # self.PC_adjust = PC_adjust
        self.PC_value_adjust = PC_value_adjust

    def transition(self, state, computation_state):
        # PC_keep, PC_adjust, PC_value_adjust = self.PC_keep, self.PC_adjust, self.PC_value_adjust
        PC_keep, PC_value_adjust = self.PC_keep, self.PC_value_adjust
        # state.program_counter = state.program_counter*PC_keep + PC_adjust + (computation_state.value1)*PC_value_adjust
        state.program_counter = state.program_counter*PC_keep + (computation_state.value1)*PC_value_adjust

class RegionBranch:
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

class RegionRewire:
    def __init__(self, value2_keep=1, value2_from_bit6=0):
        # self.value1_keep = value1_keep
        self.value2_keep = value2_keep
        self.value2_from_bit6 = value2_from_bit6

    def transition(self, state, computation_state):
        # new_value1 = computation_state.value1*self.value1_keep + ((computation_state.value1>>7) & 0x01)*self.value1_from_bit7
        new_value2 = computation_state.value2*self.value2_keep + ((computation_state.value1>>6) & 0x01)*self.value2_from_bit6
        # computation_state.value1 = new_value1
        computation_state.value2 = new_value2


class Region6:
    def __init__(self):
        pass

    def transition(self, state, computation_state):
        state.memory[computation_state.address] = computation_state.value
        

class Region7:
    def __init__(self, A_keep=1, A_adjust=0, A_value_adjust=0,
                       X_keep=1, X_adjust=0, X_value_adjust=0,
                       Y_keep=1, Y_adjust=0, Y_value_adjust=0):
        self.A_keep, self.A_adjust, self.A_value_adjust = A_keep, A_adjust, A_value_adjust
        self.X_keep, self.X_adjust, self.X_value_adjust = X_keep, X_adjust, X_value_adjust
        self.Y_keep, self.Y_adjust, self.Y_value_adjust = Y_keep, Y_adjust, Y_value_adjust

    def transition(self, state, computation_state):
        A_keep, A_adjust, A_value_adjust = self.A_keep, self.A_adjust, self.A_value_adjust
        X_keep, X_adjust, X_value_adjust = self.X_keep, self.X_adjust, self.X_value_adjust
        Y_keep, Y_adjust, Y_value_adjust = self.Y_keep, self.Y_adjust, self.Y_value_adjust

        state.A = state.A*A_keep + A_adjust + (computation_state.value)*A_value_adjust
        state.X = state.X*X_keep + X_adjust + (computation_state.value)*X_value_adjust
        state.Y = state.Y*Y_keep + Y_adjust + (computation_state.value)*Y_value_adjust

class RegionComposition:
    def __init__(self, region1=Region1(),
                       region7=Region7(),
                       boolean_logic=RegionBooleanLogic(),
                       rewire=RegionRewire(),
                       region_branch=RegionBranch(),
                       flags=Region_Flags(),
                       region6=Region6()):
        self.region1 = region1
        self.region7 = region7
        self.boolean_logic = boolean_logic
        self.rewire = rewire
        self.region_branch = region_branch
        self.flags = flags
        self.region6 = region6

        regions = [region1, region7, boolean_logic, rewire, region_branch, flags, region6]
        types = [Region1, Region7, RegionBooleanLogic, RegionRewire, RegionBranch, Region_Flags, Region6]
        assert all(isinstance(region, type_) for region, type_ in zip(regions, types)), None

    def transition(self, state, computation_state):
        self.region1.transition(state, computation_state)
        self.region7.transition(state, computation_state)
        self.region_branch.transition(state, computation_state)
        self.boolean_logic.transition(state, computation_state)
        self.rewire.transition(state, computation_state)
        self.flags.transition(state, computation_state)
