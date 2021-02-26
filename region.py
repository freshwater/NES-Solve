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

flag = lambda x: "0xFF" if 1 == x else "0x00"
flag16 = lambda x: "0xFFFF" if 1 == x else "0x0000"
signed_int = lambda x: x
unsigned_int = lambda x: x

class Region:
    def struct_form(self):
        class_name = self.__class__.__name__

        import inspect

        parameters = [v for k, v in inspect.signature(self.__init__).parameters.items() if k in self.args_OK()]
        for param in parameters:
            if param.annotation == inspect._empty:
                print("ANNOT", param.annotation, param.name)

        args = ', '.join(f'.{p.name}={p.annotation(self.__getattribute__(p.name))}' for p in parameters)
        result = ' '.join(f'''{class_name}{{{args}}}'''.split())

        return result

class Region_Wire(Region):
    def __init__(self, 
                 value1_from_address=0,
                 value1_from_data1: flag = 0,
                 value1_from_zeropage_dereference: flag16 = 0,
                 value1_from_absolute_dereference: flag16 = 0,
                 value1_from_zeropage_x_dereference: flag16 = 0,
                 value1_from_absolute_x_dereference: flag16 = 0,
                 value1_from_indirect_x_dereference: flag16 = 0,
                 value1_from_zeropage_y_dereference: flag16 = 0,
                 value1_from_absolute_y_dereference: flag16 = 0,
                 value1_from_indirect_y_dereference: flag16 = 0,
                 value1_from_stack_offset: flag = 0,
                 value1_from_A: flag = 0,
                 value1_from_X: flag = 0,
                 value1_from_Y: flag = 0,
                 address_from_absolute: flag16 = 0,
                 address_from_absolute_y: flag16 = 0,
                 address_from_absolute_dereference: flag16 = 0,
                 address_from_zeropage: flag16 = 0,
                 address_from_zeropage_x: flag16 = 0,
                 address_from_absolute_x: flag16 = 0,
                 address_from_indirect_x: flag16 = 0,
                 address_from_zeropage_y: flag16 = 0,
                 address_from_indirect_y: flag16 = 0,
                 cycle_base_increment: unsigned_int = None):
        self.value1_from_A = value1_from_A
        self.value1_from_X = value1_from_X
        self.value1_from_Y = value1_from_Y
        self.value1_from_address = value1_from_address
        self.value1_from_data1 = value1_from_data1
        self.value1_from_zeropage_dereference = value1_from_zeropage_dereference
        self.value1_from_stack_offset = value1_from_stack_offset
        self.value1_from_absolute_dereference = value1_from_absolute_dereference
        self.value1_from_indirect_x_dereference = value1_from_indirect_x_dereference
        self.value1_from_zeropage_x_dereference = value1_from_zeropage_x_dereference
        self.value1_from_zeropage_y_dereference = value1_from_zeropage_y_dereference
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
        self.address_from_absolute_dereference = address_from_absolute_dereference
        self.address_from_absolute_x = address_from_absolute_x

        self.cycle_base_increment = cycle_base_increment

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
                                    (computation_state.data1)*self.value1_from_data1 +
                                    Region_Wire.read_if(state, computation_state.data1, self.value1_from_zeropage_dereference) +
                                    (state.stack_offset)*self.value1_from_stack_offset +
                                    Region_Wire.read_if(state, computation_state.data2*0x0100 + computation_state.data1, self.value1_from_absolute_dereference) +
                                    Region_Wire.indirect_x_dereference(state, computation_state, self.value1_from_indirect_x_dereference) +
                                    Region_Wire.zeropage_x_dereference(state, computation_state, self.value1_from_zeropage_x_dereference) +
                                    Region_Wire.zeropage_y_dereference(state, computation_state, self.value1_from_zeropage_y_dereference) +
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
                                     (Region_Wire.absolute_address_indirect(state, computation_state, self.address_from_absolute_dereference))*self.address_from_absolute_dereference +
                                     (Region_Wire.absolute_x_address(state, computation_state, self.address_from_absolute_x))*self.address_from_absolute_x)

    def args_OK(self):
        return ['value1_from_data1',
                'value1_from_zeropage_dereference',
                'value1_from_zeropage_x_dereference',
                'value1_from_absolute_x_dereference',
                'value1_from_absolute_dereference',
                'value1_from_absolute_y_dereference',
                'value1_from_indirect_x_dereference',
                'value1_from_zeropage_y_dereference',
                'value1_from_indirect_y_dereference',
                'value1_from_stack_offset',
                'value1_from_A', 'value1_from_X', 'value1_from_Y',
                'address_from_absolute',
                'address_from_absolute_y',
                'address_from_absolute_dereference', 'address_from_zeropage',
                'address_from_zeropage_x',
                'address_from_absolute_x',
                'address_from_indirect_x',
                'address_from_zeropage_y',
                'address_from_indirect_y',
                'cycle_base_increment']

class Region_Compare(Region):
    def __init__(self, A_compare_with_value1: flag = 0,
                       X_compare_with_value1: flag = 0,
                       Y_compare_with_value1: flag = 0,
                       value1_out: flag = 0,
                       value3_from_carry: flag = 0):
        self.A_compare_with_value1 = A_compare_with_value1
        self.X_compare_with_value1 = X_compare_with_value1
        self.Y_compare_with_value1 = Y_compare_with_value1

        self.value1_out = value1_out
        self.value3_from_carry = value3_from_carry

    def transition(self, state, computation_state):
        result = state.A*self.A_compare_with_value1 + state.X*self.X_compare_with_value1 + state.Y*self.Y_compare_with_value1 - computation_state.value1
        carry = computation_state.value1 <= (state.A*self.A_compare_with_value1 + state.X*self.X_compare_with_value1 + state.Y*self.Y_compare_with_value1)

        computation_state.value1 = computation_state.value1*(1-self.value1_out) + result*self.value1_out
        computation_state.value3 = computation_state.value3*(1-self.value3_from_carry) + carry*self.value3_from_carry

    def args_OK(self):
        return ['A_compare_with_value1', 'X_compare_with_value1', 'Y_compare_with_value1', 'value1_out', 'value3_from_carry']

class Region_BooleanLogic(Region):
    def __init__(self, A_AND_value1: flag = 0,
                       A_OR_value1: flag = 0,
                       A_XOR_value1: flag = 0,
                       A_wire=0,
                       value1_out: flag = 0,
                       value2_wire=0,
                       value3_out: flag = 0):
        self.A_wire = A_wire
        self.A_OR_value1, self.A_XOR_value1, self.A_AND_value1 = A_OR_value1, A_XOR_value1, A_AND_value1
        self.value1_out, self.value2_wire, self.value3_out = value1_out, value2_wire, value3_out

    def transition(self, state, computation_state):
        result = byte((computation_state.value1 | state.A)*self.A_OR_value1 +
                      (computation_state.value1 ^ state.A)*self.A_XOR_value1 +
                      (computation_state.value1 & state.A)*self.A_AND_value1)

        state.A = state.A*(1-self.A_wire) + result*self.A_wire
        computation_state.value1 = computation_state.value1*(1-self.value1_out) + result*self.value1_out
        computation_state.value2 = computation_state.value2*(1-self.value2_wire) + result*self.value2_wire
        computation_state.value3 = computation_state.value3*(1-self.value3_out) + result*self.value3_out

    def args_OK(self):
        return ['A_AND_value1', 'A_OR_value1', 'A_XOR_value1', 'value1_out', 'value3_out']

class Region_BitShift(Region):
    def __init__(self, left_shift_from_value1: flag = 0,
                       right_shift_from_value1: flag = 0,
                       left_rotate_from_value1: flag = 0,
                       right_rotate_from_value1: flag = 0,
                       value1_out: flag = 0,
                       value3_from_carry: flag = 0):
        self.right_shift_from_value1 = right_shift_from_value1
        self.right_rotate_from_value1 = right_rotate_from_value1
        self.left_shift_from_value1 = left_shift_from_value1
        self.left_rotate_from_value1 = left_rotate_from_value1

        self.value1_out = value1_out
        self.value3_from_carry = value3_from_carry

    def transition(self, state, computation_state):
        any_ = self.right_shift_from_value1 + self.right_rotate_from_value1 + self.left_shift_from_value1 + self.left_rotate_from_value1
        any_right = self.right_shift_from_value1 + self.right_rotate_from_value1
        any_left = self.left_shift_from_value1 + self.left_rotate_from_value1
        any_rotate = self.left_rotate_from_value1 + self.right_rotate_from_value1

        new_ = byte(computation_state.value1*(1-any_) + ((computation_state.value1 >> 1)*any_right +
                                                         (computation_state.value1 << 1)*any_left))

        new_carry = (computation_state.value1 & 0x01)*any_right + (computation_state.value1 >> 7)*any_left
        new_ = new_*(1-any_rotate) + (((state.C << 7) | new_)*self.right_rotate_from_value1 +
                                      ((state.C << 0) | new_)*self.left_rotate_from_value1)

        computation_state.value1 = (computation_state.value1)*(1-self.value1_out) + (new_)*self.value1_out
        computation_state.value3 = (computation_state.value3)*(1-self.value3_from_carry) + (new_carry)*self.value3_from_carry

    def args_OK(self):
        return ['right_shift_from_value1', 'left_shift_from_value1',
                'right_rotate_from_value1', 'left_rotate_from_value1',
                'value1_out', 'value3_from_carry']

class Region_Arithmetic(Region):
    def __init__(self, value1_increment: signed_int = 0):
        self.value1_increment = value1_increment

    def transition(self, state, computation_state):
        computation_state.value1 = byte((computation_state.value1) + self.value1_increment)

    def args_OK(self):
        return ['value1_increment']

class Region_StackRead(Region):
    def __init__(self, value1_from_stack_read: flag16 = 0,
                       read_special_status_bits: flag = 0,
                       stack_offset_pre_adjust: signed_int = 0):
        self.value1_from_stack_read = value1_from_stack_read
        self.read_special_status_bits = read_special_status_bits
        self.stack_offset_pre_adjust = stack_offset_pre_adjust

    def transition(self, state, computation_state):
        state.stack_offset += self.stack_offset_pre_adjust
        computation_state.value1 = computation_state.value1*(1-self.value1_from_stack_read) + (state.memory[STACK_ZERO + state.stack_offset])*self.value1_from_stack_read
        special_status_bits = behaviors.Behaviors.read_special_status_bits_on_pull(state, computation_state.value1, is_PLP_or_RTI=self.read_special_status_bits)
        computation_state.value1 = computation_state.value1*(1-self.read_special_status_bits) + (special_status_bits)*self.read_special_status_bits

    def args_OK(self):
        return ['value1_from_stack_read', 'read_special_status_bits', 'stack_offset_pre_adjust']

class Region_ADC_SBC(Region):
    def __init__(self, value1_from_ADC: flag = 0,
                       value1_from_SBC: flag = 0,
                       value2_from_overflow: flag = 0,
                       value3_from_carry: flag = 0):
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

    def args_OK(self):
        return ['value1_from_ADC', 'value1_from_SBC', 'value2_from_overflow', 'value3_from_carry']

class Region_JSR_RTS_RTI(Region):
    def __init__(self, jsr_OK: flag16 = 0,
                       rts_OK: flag16 = 0,
                       rti_OK: flag16 = 0):
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

        # state.stack_offset += self.rti_OK
        # read_address = (ComputationState.NULL_ADDRESS)*(1-self.rti_OK) + (STACK_ZERO + state.stack_offset)*self.rti_OK
        # new_status = (state.memory[read_address])*self.rti_OK
        # new_status = behaviors.Behaviors.read_special_status_bits_on_pull(state, new_status, is_PLP_or_RTI=self.rti_OK)
        # new_status = (state.status_register_byte())*(1-self.rti_OK) + (new_status)*self.rti_OK
        # state.status_register_byte_set(new_status)

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

    def args_OK(self):
        return ['jsr_OK', 'rts_OK', 'rti_OK']

class Region_Branch(Region):
    def __init__(self, flag_match: unsigned_int =0,
                       N_flag_branch: flag = 0,
                       O_flag_branch: flag = 0,
                       Z_flag_branch: flag = 0,
                       C_flag_branch: flag = 0):
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

    def args_OK(self):
        return ['flag_match', 'N_flag_branch', 'O_flag_branch', 'Z_flag_branch', 'C_flag_branch']

class Region_Rewire(Region):
    def __init__(self, value1_from_status_push_bits: flag = 0,
                       value2_from_value1_bit6: flag = 0,
                       A_from_value1: flag = 0,
                       X_from_value1: flag = 0,
                       Y_from_value1: flag = 0,
                       program_counter_from_address: flag16 = 0,
                       stack_offset_from_X: flag = 0,
                       value1_keep=1,
                       value1_from_A=0,
                       value1_from_X=0,
                       value1_from_Y=0,
                       value1_from_stack_offset=0,
                       value2_keep=1,
                       A_from_X=0,
                       A_from_Y=0,
                       X_from_A=0,
                       X_from_stack_offset=0,
                       Y_from_A=0):
        self.value1_keep = value1_keep
        self.value1_from_A = value1_from_A
        self.value1_from_X = value1_from_X
        self.value1_from_Y = value1_from_Y
        self.value1_from_status_push_bits = value1_from_status_push_bits
        self.value1_from_stack_offset = value1_from_stack_offset

        self.value2_keep = value2_keep
        self.value2_from_value1_bit6 = value2_from_value1_bit6

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
                                                                  (special_status_bits)*self.value1_from_status_push_bits +
                                                                  (state.stack_offset)*self.value1_from_stack_offset)

        new_value2 = computation_state.value2*self.value2_keep + ((computation_state.value1>>6) & 0x01)*self.value2_from_value1_bit6

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

    def args_OK(self):
        return ['value1_from_status_push_bits',
                'value2_from_value1_bit6',
                'A_from_value1', 'X_from_value1', 'Y_from_value1',
                'program_counter_from_address',
                'stack_offset_from_X']

class Region_ImplementationState(Region):
    def __init__(self, store_write_from_value1: flag = 0,
                       value1_read_from_store: flag = 0):
        self.store_write_from_value1 = store_write_from_value1
        self.value1_read_from_store = value1_read_from_store

    def args_OK(self):
        return ['store_write_from_value1',
                'value1_read_from_store']


class Region_Flags(Region):
    def __init__(self, N_keep: flag = 1, N_adjust: unsigned_int = 0, N_adjust_source: unsigned_int = Wire.NULL,
                       O_keep: flag = 1, O_adjust: unsigned_int = 0, O_adjust_direct: unsigned_int = Wire.NULL,
                       U_keep=1, U_adjust=0,
                       B_keep=1, B_adjust=0,
                       D_keep: flag = 1, D_adjust: unsigned_int = 0,
                       I_keep: flag = 1, I_adjust: unsigned_int = 0,
                       Z_keep: flag = 1, Z_adjust: unsigned_int = 0, Z_adjust_source: unsigned_int = Wire.NULL,
                       C_keep: flag = 1, C_adjust: unsigned_int = 0, C_adjust_direct: unsigned_int = Wire.NULL,
                       set_byte_from_value1: flag = 0):

        self.N_keep, self.N_adjust = N_keep, N_adjust
        self.O_keep, self.O_adjust = O_keep, O_adjust
        self.U_keep, self.U_adjust = U_keep, U_adjust
        self.B_keep, self.B_adjust = B_keep, B_adjust
        self.D_keep, self.D_adjust = D_keep, D_adjust
        self.I_keep, self.I_adjust = I_keep, I_adjust
        self.Z_keep, self.Z_adjust = Z_keep, Z_adjust
        self.C_keep, self.C_adjust = C_keep, C_adjust

        self.N_adjust_source = N_adjust_source
        self.O_adjust_direct = O_adjust_direct
        self.Z_adjust_source = Z_adjust_source
        self.C_adjust_direct = C_adjust_direct

        self.set_byte_from_value1 = set_byte_from_value1

    def transition(self, state, computation_state):
        N_value = ((self.N_adjust_source == 1)*computation_state.value1 +
                   (self.N_adjust_source == 2)*computation_state.value2 +
                   (self.N_adjust_source == 3)*computation_state.value3)

        O_value = ((self.O_adjust_direct == 1)*computation_state.value1 +
                   (self.O_adjust_direct == 2)*computation_state.value2 +
                   (self.O_adjust_direct == 3)*computation_state.value3)

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

        byte = state.status_register_byte()
        new_value = byte*(1-self.set_byte_from_value1) + computation_state.value1*self.set_byte_from_value1
        state.status_register_byte_set(new_value)

    def args_OK(self):
        return ['N_keep', 'N_adjust', 'N_adjust_source',
                'O_keep', 'O_adjust', 'O_adjust_direct',
                'D_keep', 'D_adjust',
                'I_keep', 'I_adjust',
                'Z_keep', 'Z_adjust', 'Z_adjust_source',
                'C_keep', 'C_adjust', 'C_adjust_direct',
                'set_byte_from_value1']

class Region_Write(Region):
    def __init__(self, memory_write_value1: flag16 = 0):
        self.memory_write_value1 = memory_write_value1

    def transition(self, state, computation_state):
        address = ComputationState.NULL_ADDRESS*(1-self.memory_write_value1) + computation_state.address*self.memory_write_value1
        state.memory[address] = computation_state.value1

    def args_OK(self):
        return ['memory_write_value1']

class Region_StackWrite(Region):
    def __init__(self, stack_write_value1: flag16 = 0,
                       stack_offset_post_adjust: signed_int = 0):
        self.stack_write_value1 = stack_write_value1
        self.stack_offset_post_adjust = stack_offset_post_adjust

    def transition(self, state, computation_state):
        address = ComputationState.NULL_ADDRESS*(1-self.stack_write_value1) + (STACK_ZERO + state.stack_offset)*self.stack_write_value1
        state.memory[address] = computation_state.value1

        state.stack_offset += self.stack_offset_post_adjust

    def args_OK(self):
        return ['stack_write_value1', 'stack_offset_post_adjust']

class Region_ProgramCounter(Region):
    def __init__(self, PC_increment: unsigned_int = 0):
        self.PC_increment = PC_increment

    def args_OK(self):
        return ['PC_increment']

class RegionComposition:
    def __init__(self, compare=Region_Compare(),
                       stack_read=Region_StackRead(),
                       boolean_logic=Region_BooleanLogic(),
                       arithmetic=Region_Arithmetic(),
                       adc_sbc=Region_ADC_SBC(),
                       jsr_rts_rti=Region_JSR_RTS_RTI(),
                       bit_shift=Region_BitShift(),
                       rewire=Region_Rewire(),
                       branch=Region_Branch(),
                       stack_write=Region_StackWrite(),
                       implementation_state=Region_ImplementationState(),
                       flags=Region_Flags(),
                       write=Region_Write()):

        self.regions = [
            # (wire, Region_Wire),
            ("compare", compare, Region_Compare),
            ("boolean_logic", boolean_logic, Region_BooleanLogic),
            ("bit_shift", bit_shift, Region_BitShift),
            ("arithmetic", arithmetic, Region_Arithmetic),
            ("stack_read", stack_read, Region_StackRead),
            ("adc_sbc", adc_sbc, Region_ADC_SBC),
            ("jsr_rts_rti", jsr_rts_rti, Region_JSR_RTS_RTI),
            ("branch", branch, Region_Branch),
            ("rewire", rewire, Region_Rewire),
            ("implementation_state", implementation_state, Region_ImplementationState),
            ("flags", flags, Region_Flags),
            ("write", write, Region_Write),
            ("stack_write", stack_write, Region_StackWrite),
        ]

        assert all(isinstance(region, type_) for name, region, type_ in self.regions), None

    def transition(self, state, computation_state):
        for _name, region, _class in self.regions:
            region.transition(state, computation_state)

    def struct(self, wire_region, byte_count):
        regions_OK = [
            "wire",
            "stack_read",
            "boolean_logic",
            "arithmetic",
            "adc_sbc",
            "jsr_rts_rti",
            "rewire",
            "branch",
            "write",
            "stack_write",
            "implementation_state",
            "flags",
            "program_counter",
            "compare",
            "bit_shift"
        ]

        regions = [(name, region) for name, region, _class in self.regions if name in regions_OK]
        regions = [("wire", wire_region)] + regions + [("program_counter", Region_ProgramCounter(PC_increment=byte_count))]

        return 'RegionComposition{' + ', '.join(f'.{name}={region.struct_form()}' for name, region in regions) + '}'
