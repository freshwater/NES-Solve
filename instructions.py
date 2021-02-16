
import numpy as np
from behaviors import Behaviors

from region import *

#-

def byte(data):
    return data & 0x00FF

#-

STACK_ZERO = 0x0100

#-

def implied(): # state, _data1, _data2):
    # pass
    return Region_Wire()

def immediate(): # state, data1, _data2):
    # return data1
    return Region_Wire(value1_from_zeropage=1)

def absolute_dereference(): # state, data1, data2):
    # return state.memory[data2*0x0100 + data1]
    return Region_Wire(value1_from_absolute_dereference=1)

def absolute_address(): # state, data1, data2):
    # return data2*0x0100 + data1
    return Region_Wire(address_from_absolute=1)

def absolute():
    return Region_Wire(value1_from_absolute_dereference=1, address_from_absolute=1)

def absolute_address_dereference(): # state, data1, data2):
    # indirect wraps at page boundary
    # L = state.memory[data2*0x0100 + data1]
    # H = state.memory[data2*0x0100 + byte(data1 + 1)]
    # return H*0x0100 + L
    return Region_Wire(address_from_absolute_indirect=1)

def zeropage_dereference(): # state, data1, _data2):
    # return state.memory[data1]
    return Region_Wire(value1_from_zeropage_dereference=1, address_from_zeropage=1)

def zeropage_address(): # state, data1, _data2):
    # return data1
    return Region_Wire(address_from_zeropage=1)

def relative_address(): # state, data1, _data2):
    # return data1
    return Region_Wire(value1_from_zeropage=1)


# X

def absolute_x_dereference(): # state, data1, data2):
    # return state.memory[data2*0x0100 + data1 + state.X]
    return Region_Wire(value1_from_absolute_x_dereference=1)

def absolute_x_address(): # state, data1, data2):
    # return data2*0x0100 + data1 + state.X
    return Region_Wire(address_from_absolute_x=1)

def absolute_x(): # state, data1, data2):
    # return data2*0x0100 + data1 + state.X
    return Region_Wire(value1_from_absolute_x_dereference=1, address_from_absolute_x=1)

def zeropage_x_dereference(): # state, data1, _data2):
    # return state.memory[byte(data1 + state.X)]
    return Region_Wire(value1_from_zeropage_x=1)

def zeropage_x_address(): # state, data1, _data2):
    # return byte(data1 + state.X)
    return Region_Wire(address_from_zeropage_x=1)

def zeropage_x():
    return Region_Wire(value1_from_zeropage_x=1, address_from_zeropage_x=1)

def indirect_x_dereference(): # state, data1, _data2):
    # L = state.memory[byte(data1 + state.X)]
    # H = state.memory[byte(data1 + state.X + 1)]
    # address = H*0x0100 + L
    # return state.memory[address]
    return Region_Wire(value1_from_indirect_x_dereference=1)

def indirect_x_address(): # state, data1, _data2):
    # L = state.memory[byte(data1 + state.X)]
    # H = state.memory[byte(data1 + state.X + 1)]
    # return H*0x0100 + L
    return Region_Wire(address_from_indirect_x=1)


# Y

def absolute_y_dereference(): # state, data1, data2):
    # address = data2*0x0100 + data1
    # return state.memory[address + state.Y]
    return Region_Wire(value1_from_absolute_y_dereference=1)

def absolute_y_address(): # state, data1, data2):
    # address = data2*0x0100 + data1
    # return address + state.Y
    return Region_Wire(address_from_absolute_y=1)

def zeropage_y_dereference(): # state, data1, _data2):
    # return state.memory[byte(data1 + state.Y)]
    return Region_Wire(value1_from_zeropage_y=1)

def zeropage_y_address(): # state, data1, _data2):
    # return byte(data1 + state.Y)
    return Region_Wire(address_from_zeropage_y=1)

def indirect_y_dereference(): # state, data1, _data2):
    # address = state.memory[byte(data1+1)]*0x0100 + state.memory[data1]
    # return state.memory[address + state.Y]
    return Region_Wire(value1_from_indirect_y_dereference=1)

def indirect_y_address(): # state, data1, _data2):
    # address = state.memory[data1+1]*0x0100 + state.memory[data1]
    # return address + state.Y
    return Region_Wire(address_from_indirect_y=1)

#-

byte_counts = {implied: 1, immediate: 2, relative_address: 2, zeropage_dereference: 2,
               zeropage_address: 2, zeropage_x_dereference: 2, zeropage_x_address: 2,
               zeropage_y_dereference: 2, zeropage_y_address: 2, absolute_dereference: 3,
               absolute: 3, zeropage_x: 2, absolute_x: 3,
               absolute_address: 3, absolute_x_dereference: 3, absolute_x_address: 3,
               indirect_x_dereference: 2, indirect_x_address: 2, indirect_y_address: 2,
               indirect_y_dereference: 2, absolute_address_dereference: 3,
               absolute_y_dereference: 3, absolute_y_address: 3}
#-

def SEI() -> [(0x78, implied)]:
    # state.status_register['Interrupt'] = 1
    return RegionComposition(flags=Region_Flags(I_keep=0, I_adjust=1))
def CLI() -> [(0x58, implied)]:
    # state.status_register['Interrupt'] = 0
    Region_Flags.transition(state, ComputationState(), I_keep=0, I_adjust=0)
def CLD() -> [(0xD8, implied)]:
    # state.status_register['Decimal'] = 0
    return RegionComposition(flags=Region_Flags(D_keep=0, D_adjust=0))
def SEC() -> [(0x38, implied)]:
    # state.status_register['Carry'] = 1
    return RegionComposition(flags=Region_Flags(C_keep=0, C_adjust=1))
def CLC() -> [(0x18, implied)]:
    # state.status_register['Carry'] = 0
    return RegionComposition(flags=Region_Flags(C_keep=0, C_adjust=0))
def SED() -> [(0xF8, implied)]:
    # state.status_register['Decimal'] = 1
    return RegionComposition(flags=Region_Flags(D_keep=0, D_adjust=1))
def CLV() -> [(0xB8, implied)]:
    # state.status_register['Overflow'] = 0
    return RegionComposition(flags=Region_Flags(O_keep=0, O_adjust=0))

def NOP() -> [(0x04, zeropage_address), (0x0C, absolute_address), (0x14, zeropage_x_address), (0x1A, implied),
                      (0x1C, absolute_x_address), (0x34, zeropage_x_address), (0x3A, implied), (0x3C, absolute_x_address),
                      (0x44, zeropage_address), (0x54, zeropage_x_address), (0x5A, implied), (0x5C, absolute_x_address),
                      (0x64, zeropage_address), (0x74, zeropage_x_address), (0x7A, implied), (0x7C, absolute_x_address),
                      (0x80, immediate), (0xD4, zeropage_x_address), (0xDA, implied), (0xDC, absolute_x_address), (0xEA, implied),
                      (0xF4, zeropage_x_address), (0xFA, implied), (0xFC, absolute_x_address)]:
    return RegionComposition()

def LDA() -> [(0xA1, indirect_x_dereference), (0xA5, zeropage_dereference), (0xA9, immediate),
                      (0xAD, absolute), (0xB1, indirect_y_dereference), (0xB5, zeropage_x_dereference),
                      (0xB9, absolute_y_dereference), (0xBD, absolute_x_dereference)]:
    # state.A = a; Z_set(state, a); N_set(state, a)
    return RegionComposition(
        rewire=Region_Rewire(A_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def LDX() -> [(0xA2, immediate), (0xA6, zeropage_dereference), (0xAE, absolute_dereference),
                      (0xB6, zeropage_y_dereference), (0xBE, absolute_y_dereference)]:
    # state.X = a; Z_set(state, a); N_set(state, a)
    return RegionComposition(
        rewire=Region_Rewire(X_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def LDY() -> [(0xA0, immediate), (0xA4, zeropage_dereference), (0xAC, absolute_dereference),
                      (0xB4, zeropage_x_dereference), (0xBC, absolute_x_dereference)]:
    # state.Y = a; Z_set(state, a); N_set(state, a)
    return RegionComposition(
        rewire=Region_Rewire(Y_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def DEX() -> [(0xCA, implied, Region_Wire(value1_from_X=1))]:
    # state.X = byte(state.X - 1); Z_set(state, state.X); N_set(state, state.X)
    return RegionComposition(
        arithmetic=Region_Arithmetic(value1_increment=-1),
        rewire=Region_Rewire(X_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def DEY() -> [(0x88, implied, Region_Wire(value1_from_Y=1))]:
    # state.Y = byte(state.Y - 1); Z_set(state, state.Y); N_set(state, state.Y)
    return RegionComposition(
        arithmetic=Region_Arithmetic(value1_increment=-1),
        rewire=Region_Rewire(Y_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def INY() -> [(0xC8, implied, Region_Wire(value1_from_Y=1))]:
    # state.Y = byte(state.Y + 1); Z_set(state, state.Y); N_set(state, state.Y)
    return RegionComposition(
        arithmetic=Region_Arithmetic(value1_increment=1),
        rewire=Region_Rewire(Y_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def INC() -> [(0xE6, zeropage_dereference), (0xEE, absolute), (0xF6, zeropage_x),
                      (0xFE, absolute_x)]:
    # state.memory[a] += 1; Z_set(state, state.memory[a]); N_set(state, state.memory[a])
    return RegionComposition(
        arithmetic=Region_Arithmetic(value1_increment=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1),
        write=Region_Write(address_write=1))

def INX() -> [(0xE8, implied, Region_Wire(value1_from_X=1))]:
    # state.X = byte(state.X + 1); Z_set(state, state.X); N_set(state, state.X)
    return RegionComposition(
        arithmetic=Region_Arithmetic(value1_increment=1),
        rewire=Region_Rewire(X_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def DEC_zpg() -> [(0xC6, zeropage_dereference), (0xCE, absolute), (0xD6, zeropage_x),
                          (0xDE, absolute_x)]:
    # state.memory[a] -= 1
    # Z_set(state, state.memory[a])
    # N_set(state, state.memory[a])
    return RegionComposition(
        arithmetic=Region_Arithmetic(value1_increment=-1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1),
        write=Region_Write(address_write=1))

def BIT() -> [(0x2C, absolute_dereference)]:
    # result = byte(state.A & a)
    # Z_set(state, result)
    # N_set(state, a)
    # state.status_register['Overflow'] = (a >> 6) & 0x01
    return RegionComposition(
        boolean_logic=Region_BooleanLogic(AND_A=1, value3_wire=1),
        rewire=Region_Rewire(value2_from_bit6=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           O_keep=0, O_adjust=0, O_adjust_source=Wire.VALUE2,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE3))

def BIT_zpg() -> [(0x24, zeropage_dereference)]:
    # state.status_register['Negative'] = (a >> 7) & 0x01
    # state.status_register['Overflow'] = (a >> 6) & 0x01
    # Z_set(state, byte(state.A & a))
    return RegionComposition(
        boolean_logic=Region_BooleanLogic(AND_A=1, A_wire=0, value3_wire=1),
        rewire=Region_Rewire(value2_keep=0, value2_from_bit6=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           O_keep=0, O_adjust=0, O_adjust_source=Wire.VALUE2,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE3))

def CMP() -> [(0xC1, indirect_x_dereference), (0xC5, zeropage_dereference), (0xC9, immediate),
                      (0xCD, absolute_dereference), (0xD1, indirect_y_dereference), (0xD5, zeropage_x_dereference),
                      (0xD9, absolute_y_dereference), (0xDD, absolute_x_dereference)]:
    # result = byte(state.A - a)
    # Z_set(state, result); N_set(state, result)
    # if a <= state.A:
    #     state.status_register['Carry'] = 1
    # else:
    #     state.status_register['Carry'] = 0
    return RegionComposition(
        compare=Region_Compare(A_compare=1, value1_out=1, value3_from_carry=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3))

def CPX() -> [(0xE0, immediate), (0xE4, zeropage_dereference), (0xEC, absolute_dereference)]:
    # result = state.X - a
    # Z_set(state, result); N_set(state, result)
    # if a <= state.X:
    #     state.status_register['Carry'] = 1
    # else:
    #     state.status_register['Carry'] = 0
    return RegionComposition(
        compare=Region_Compare(X_compare=1, value1_out=1, value3_from_carry=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3))

def CPY() -> [(0xC0, immediate), (0xC4, zeropage_dereference), (0xCC, absolute_dereference)]:
    # result = byte(state.Y - a)
    # Z_set(state, result); N_set(state, result)
    # if a <= state.Y:
    #     state.status_register['Carry'] = 1
    # else:
    #     state.status_register['Carry'] = 0
    return RegionComposition(
        compare=Region_Compare(Y_compare=1, value1_out=1, value3_from_carry=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3))

def ORA() -> [(0x01, indirect_x_dereference), (0x05, zeropage_dereference), (0x09, immediate),
                      (0x0D, absolute_dereference), (0x11, indirect_y_dereference), (0x15, zeropage_x_dereference),
                      (0x19, absolute_y_dereference), (0x1D, absolute_x_dereference)]:
    # state.A |= a; Z_set(state, state.A); N_set(state, state.A)
    return RegionComposition(
        boolean_logic=Region_BooleanLogic(OR_A=1, value1_wire=1),
        rewire=Region_Rewire(A_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def EOR() -> [(0x41, indirect_x_dereference), (0x45, zeropage_dereference), (0x49, immediate),
                      (0x4D, absolute_dereference), (0x51, indirect_y_dereference), (0x55, zeropage_x_dereference),
                      (0x59, absolute_y_dereference), (0x5D, absolute_x_dereference)]:
    # state.A ^= a; Z_set(state, state.A); N_set(state, state.A)
    return RegionComposition(
        boolean_logic=Region_BooleanLogic(XOR_A=1, value1_wire=1),
        rewire=Region_Rewire(A_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def AND() -> [(0x21, indirect_x_dereference), (0x25, zeropage_dereference), (0x29, immediate),
                      (0x2D, absolute_dereference), (0x31, indirect_y_dereference), (0x35, zeropage_x_dereference),
                      (0x39, absolute_y_dereference), (0x3D, absolute_x_dereference)]:
    # state.A &= a; Z_set(state, state.A); N_set(state, state.A)
    return RegionComposition(
        boolean_logic=Region_BooleanLogic(AND_A=1, value1_wire=1),
        rewire=Region_Rewire(A_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def ASL() -> [(0x0A, implied, Region_Wire(value1_from_A=1))]:
    # result = state.A << 1
    # Z_set(state, byte(result))
    # N_set(state, byte(result))
    # state.status_register['Carry'] = 1*(result & 0xFF00 > 0)
    # state.A = byte(result)
    return RegionComposition(
        # wire=Region_Wire(value1_from_A=1),
        bit_shift=Region_BitShift(left_shift=1, value1_out=1, value3_from_carry=1),
        rewire=Region_Rewire(A_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3))

def ASL_zpg() -> [(0x06, zeropage_dereference), (0x0E, absolute), (0x16, zeropage_x),
                          (0x1E, absolute_x)]:
    # result = byte(state.memory[a] << 1)
    # Z_set(state, result)
    # N_set(state, result)
    # state.status_register['Carry'] = state.memory[a] >> 7
    # state.memory[a] = result
    return RegionComposition(
        bit_shift=Region_BitShift(left_shift=1, value1_out=1, value3_from_carry=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3),
        write=Region_Write(address_write=1))

def LSR() -> [(0x4A, implied, Region_Wire(value1_from_A=1))]:
    # result = state.A >> 1
    # Z_set(state, result)
    # N_set(state, result)
    # state.status_register['Carry'] = state.A & 0x01
    # state.A = result
    return RegionComposition(
        # wire=Region_Wire(value1_from_A=1),
        bit_shift=Region_BitShift(right_shift=1, value1_out=1, value3_from_carry=1),
        rewire=Region_Rewire(A_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3))

def LSR_zpg() -> [(0x46, zeropage_dereference),#, Region_Wire(value1_from_zeropage_dereference=1, address_from_zeropage=1)),
                          (0x4E, absolute), (0x56, zeropage_x),
                          (0x5E, absolute_x)]:
    # result = state.memory[a] >> 1
    # Z_set(state, result)
    # N_set(state, result)
    # state.status_register['Carry'] = state.memory[a] & 0x01
    # state.memory[a] = result
    return RegionComposition(
        # wire=Region_Wire(value1_from_address=1),
        bit_shift=Region_BitShift(right_shift=1, value1_out=1, value3_from_carry=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3),
        write=Region_Write(address_write=1))

def ROL() -> [(0x2A, implied, Region_Wire(value1_from_A=1))]:
    # result = (state.A << 1) | state.status_register['Carry']
    # state.status_register['Carry'] = 1*(result & 0xFF00 > 0)
    # result = byte(result)
    # Z_set(state, result)
    # N_set(state, result)
    # state.A = result
    return RegionComposition(
        # wire=Region_Wire(value1_from_A=1),
        bit_shift=Region_BitShift(left_rotate=1, value1_out=1, value3_from_carry=1),
        rewire=Region_Rewire(A_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3))

def ROL_zpg() -> [(0x26, zeropage_dereference), (0x2E, absolute), (0x36, zeropage_x),
                          (0x3E, absolute_x)]:
    # result = (state.memory[a] << 1) | state.status_register['Carry']
    # state.status_register['Carry'] = 1*(result & 0xFF00 > 0)
    # result = byte(result)
    # Z_set(state, result)
    # N_set(state, result)
    # state.memory[a] = result
    return RegionComposition(
        bit_shift=Region_BitShift(left_rotate=1, value1_out=1, value3_from_carry=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3),
        write=Region_Write(address_write=1))

def ROR() -> [(0x6A, implied, Region_Wire(value1_from_A=1))]:
    # result = state.status_register['Carry'] << 7
    # state.status_register['Carry'] = state.A & 0x01
    # result += state.A >> 1
    # N_set(state, result)
    # Z_set(state, result)
    # state.A = result
    return RegionComposition(
        # wire=Region_Wire(value1_from_A=1),
        bit_shift=Region_BitShift(right_rotate=1, value1_out=1, value3_from_carry=1),
        rewire=Region_Rewire(A_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3))

def ROR_zpg() -> [(0x66, zeropage_dereference), (0x6E, absolute), (0x76, zeropage_x),
                          (0x7E, absolute_x)]:
    # result = state.status_register['Carry'] << 7
    # state.status_register['Carry'] = state.memory[a] & 0x01
    # result += state.memory[a] >> 1
    # N_set(state, result)
    # Z_set(state, result)
    # state.memory[a] = result
    return RegionComposition(
        # wire=Region_Wire(value1_from_address=1),
        bit_shift=Region_BitShift(right_rotate=1, value1_out=1, value3_from_carry=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3),
        write=Region_Write(address_write=1))

def ADC() -> [(0x61, indirect_x_dereference), (0x65, zeropage_dereference), (0x69, immediate),
                      (0x6D, absolute_dereference), (0x71, indirect_y_dereference), (0x75, zeropage_x_dereference),
                      (0x79, absolute_y_dereference), (0x7D, absolute_x_dereference)]:
    # result = state.A + a + state.C
    # Z_set(state, byte(result))
    # N_set(state, byte(result))
    # state.C = 1*(result & 0xFF00 > 0)
    # v1 =  ~(state.A ^ a) & (state.A ^ result) & 0x0080
    # state.O = 1*(v1 > 0)
    # state.A = byte(result)
    return RegionComposition(
        adc_sbc=Region_ADC_SBC(value1_from_ADC=1, value2_from_overflow=1, value3_from_carry=1),
        rewire=Region_Rewire(A_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           O_keep=0, O_adjust=0, O_adjust_source=Wire.VALUE2,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3))

def SBC() -> [(0xE1, indirect_x_dereference), (0xE5, zeropage_dereference), (0xE9, immediate), (0xEB, immediate),
                      (0xED, absolute_dereference), (0xF1, indirect_y_dereference), (0xF5, zeropage_x_dereference),
                      (0xF9, absolute_y_dereference), (0xFD, absolute_x_dereference)]:
    # a ^= 0xFF
    # result = state.A + a + state.C
    # Z_set(state, byte(result))
    # N_set(state, byte(result))
    # state.C = 1*(result & 0xFF00 > 0)
    # v1 =  ~(state.A ^ a) & (state.A ^ result) & 0x0080
    # state.O = 1*(v1 > 0)
    # state.A = byte(result)
    return RegionComposition(
        adc_sbc=Region_ADC_SBC(value1_from_SBC=1, value2_from_overflow=1, value3_from_carry=1),
        rewire=Region_Rewire(A_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           O_keep=0, O_adjust=0, O_adjust_source=Wire.VALUE2,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1,
                           C_keep=0, C_adjust=0, C_adjust_direct=Wire.VALUE3))

def STA() -> [(0x81, indirect_x_address), (0x8D, absolute_address), (0x85, zeropage_address),
                      (0x91, indirect_y_address), (0x95, zeropage_x_address), (0x99, absolute_y_address),
                      (0x9D, absolute_x_address)]:
    # state.memory[a] = state.A
    return RegionComposition(
        rewire=Region_Rewire(value1_from_A=1),
        write=Region_Write(address_write=1))

def STX() -> [(0x86, zeropage_address), (0x8E, absolute_address), (0x96, zeropage_y_address)]:
    # state.memory[a] = state.X
    return RegionComposition(
        rewire=Region_Rewire(value1_from_X=1),
        write=Region_Write(address_write=1))

def STY() -> [(0x84, zeropage_address), (0x8C, absolute_address), (0x94, zeropage_x_address)]:
    # state.memory[a] = state.Y
    return RegionComposition(
        rewire=Region_Rewire(value1_from_Y=1),
        write=Region_Write(address_write=1))

def TXA() -> [(0x8A, implied)]:
    # state.A = state.X; Z_set(state, state.A); N_set(state, state.A)
    return RegionComposition(
        rewire=Region_Rewire(A_from_X=1, value1_from_X=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def TYA() -> [(0x98, implied)]:
    # state.A = state.Y; Z_set(state, state.A); N_set(state, state.A)
    return RegionComposition(
        rewire=Region_Rewire(A_from_Y=1, value1_from_Y=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def TAX() -> [(0xAA, implied)]:
    # state.X = state.A; Z_set(state, state.X); N_set(state, state.X)
    return RegionComposition(
        rewire=Region_Rewire(X_from_A=1, value1_from_A=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def TAY() -> [(0xA8, implied)]:
    # state.Y = state.A; Z_set(state, state.Y); N_set(state, state.Y)
    return RegionComposition(
        rewire=Region_Rewire(Y_from_A=1, value1_from_A=1))
        #t
        # flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
        #                    Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def TSX() -> [(0xBA, implied)]:
    # state.X = state.stack_offset; Z_set(state, state.X); N_set(state, state.X)
    return RegionComposition(
        rewire=Region_Rewire(X_from_stack_offset=1, value1_from_stack_offset=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def TXS() -> [(0x9A, implied)]:
    # state.stack_offset = state.X
    return RegionComposition(rewire=Region_Rewire(stack_offset_from_X=1))

def BPL() -> [(0x10, relative_address)]:
    # if state.status_register['Negative'] == 0:
    #     state.program_counter += np.int8(a)
    return RegionComposition(branch=Region_Branch(flag_match=0, N_flag_branch=1))

def BMI() -> [(0x30, relative_address)]:
    # if state.status_register['Negative'] == 1:
    #     state.program_counter += np.int8(a)
    return RegionComposition(branch=Region_Branch(flag_match=1, N_flag_branch=1))

def BCC() -> [(0x90, relative_address)]:
    # if state.status_register['Carry'] == 0:
    #     state.program_counter += np.int8(a)
    return RegionComposition(branch=Region_Branch(flag_match=0, C_flag_branch=1))

def BCS() -> [(0xB0, relative_address)]:
    # if state.status_register['Carry'] == 1:
    #     state.program_counter += np.int8(a)
    return RegionComposition(branch=Region_Branch(flag_match=1, C_flag_branch=1))

def BNE() -> [(0xD0, relative_address)]:
    # if state.status_register['Zero'] == 0:
    #     state.program_counter += np.int8(a)
    return RegionComposition(branch=Region_Branch(flag_match=0, Z_flag_branch=1))

def BEQ() -> [(0xF0, relative_address)]:
    # if state.status_register['Zero'] == 1:
    #     state.program_counter += np.int8(a)
    return RegionComposition(branch=Region_Branch(flag_match=1, Z_flag_branch=1))

def BVC() -> [(0x50, relative_address)]:
    # if state.status_register['Overflow'] == 0:
    #     state.program_counter += np.int8(a)
    return RegionComposition(branch=Region_Branch(flag_match=0, O_flag_branch=1))

def JMP() -> [(0x4C, absolute_address), (0x6C, absolute_address_dereference)]:
    # state.program_counter = a
    return RegionComposition(rewire=Region_Rewire(program_counter_from_address=1))

def BVS() -> [(0x70, relative_address)]:
    # if state.status_register['Overflow'] == 1:
    #     state.program_counter += np.int8(a)
    return RegionComposition(branch=Region_Branch(flag_match=1, O_flag_branch=1))

def BRK() -> [(0x00, implied)]:
    assert False, True
    state.status_register['Interrupt'] = 1

    pc_H = state.program_counter >> 8
    pc_L = state.program_counter & 0x00FF
    state.memory[STACK_ZERO + state.stack_offset] = pc_H
    state.stack_offset -= 1
    state.memory[STACK_ZERO + state.stack_offset] = pc_L
    state.stack_offset -= 1

    status_register = state.status_register_byte()
    status_register = Behaviors.write_special_status_bits_on_push(BRK, status_register)
    state.memory[STACK_ZERO + state.stack_offset] = status_register
    # state.stack_offset -= 1

    state.program_counter = state.memory[0xFFFF]*0x0100 + state.memory[0xFFFE]

def NMI():
    nmi_L = state.memory[0xFFFA]
    nmi_H = state.memory[0xFFFB]
    data = nmi_H*0x0100 + nmi_L

    JSR(state, data)

def JSR() -> [(0x20, absolute_address)]:
    # Stack is from range 0x0100-0x1FF and grows down from 0x0100 + 0xFD.
    # Adjust for current program counter incrementing.
    # program_counter = state.program_counter - 1
    # pc_H = program_counter >> 8
    # pc_L = program_counter & 0x00FF
    # state.memory[STACK_ZERO + state.stack_offset] = pc_H
    # state.stack_offset -= 1
    # state.memory[STACK_ZERO + state.stack_offset] = pc_L
    # state.stack_offset -= 1
    # state.program_counter = a
    return RegionComposition(
        jsr_rts_rti=Region_JSR_RTS_RTI(jsr_OK=1))

def RTI() -> [(0x40, implied)]:
    # state.stack_offset += 1
    # status_register = state.memory[STACK_ZERO + state.stack_offset]
    # status_register = Behaviors.read_special_status_bits_on_pull(state, status_register)
    # state.status_register_byte_set(status_register)
    # state.stack_offset += 1
    # pc_L = state.memory[STACK_ZERO + state.stack_offset]
    # state.stack_offset += 1
    # pc_H = state.memory[STACK_ZERO + state.stack_offset]
    # program_counter = (pc_H <<8) + pc_L
    # state.program_counter = program_counter
    return RegionComposition(
        jsr_rts_rti=Region_JSR_RTS_RTI(rti_OK=1))

def RTS() -> [(0x60, implied)]:
    # state.stack_offset += 1
    # pc_L = state.memory[STACK_ZERO + state.stack_offset]
    # state.stack_offset += 1
    # pc_H = state.memory[STACK_ZERO + state.stack_offset]
    # program_counter = (pc_H << 8) + pc_L
    # state.program_counter = program_counter + 1
    return RegionComposition(
        jsr_rts_rti=Region_JSR_RTS_RTI(rts_OK=1))

def PHA() -> [(0x48, implied)]:
    # state.memory[STACK_ZERO + state.stack_offset] = state.A
    # state.stack_offset -= 1
    return RegionComposition(
        rewire=Region_Rewire(value1_from_A=1),
        stack_write=Region_StackWrite(write_from_value1=1),
        stack_offset2=Region_StackOffset(offset_keep=1, offset_adjust=-1))

def PHP() -> [(0x08, implied)]:
    # status_register = state.status_register_byte()
    # status_register = Behaviors.write_special_status_bits_on_push(PHP, status_register)
    # state.memory[STACK_ZERO + state.stack_offset] = status_register
    # state.stack_offset -= 1
    return RegionComposition(
        rewire=Region_Rewire(value1_from_P_push_bits=1),
        stack_write=Region_StackWrite(write_from_value1=1),
        stack_offset2=Region_StackOffset(offset_keep=1, offset_adjust=-1)
    )

def PLA() -> [(0x68, implied)]:
    # state.stack_offset += 1
    # state.A = state.memory[STACK_ZERO + state.stack_offset]
    # Z_set(state, state.A)
    # N_set(state, state.A)
    return RegionComposition(
        stack_offset1=Region_StackOffset(offset_keep=1, offset_adjust=1),
        stack_read=Region_StackRead(value1_from_read=1),
        rewire=Region_Rewire(A_from_value1=1),
        flags=Region_Flags(N_keep=0, N_adjust=0, N_adjust_source=Wire.VALUE1,
                           Z_keep=0, Z_adjust=0, Z_adjust_source=Wire.VALUE1))

def PLP() -> [(0x28, implied)]:
    # state.stack_offset += 1
    # data = state.memory[STACK_ZERO + state.stack_offset]
    # data = Behaviors.read_special_status_bits_on_pull(state, data)
    # state.status_register_byte_set(data)
    return RegionComposition(
        stack_offset1=Region_StackOffset(offset_keep=1, offset_adjust=1),
        stack_read=Region_StackRead(value1_from_read=1, read_special_status_bits=1),
        flags_byte=Region_FlagsByte(set_from_value1=1))

#-

def LAX() -> [(0xA3, indirect_x_dereference), (0xA7, zeropage_dereference), (0xAF, absolute_dereference),
                      (0xB3, indirect_y_dereference), (0xB7, zeropage_y_dereference), (0xBF, absolute_y_dereference)]:
    assert None, "The End"
    LDA(state, a)
    TAX(state, a)

def SAX() -> [(0x83, indirect_x_address), (0x87, zeropage_address), (0x8F, absolute_address),
                      (0x97, zeropage_y_address)]:
    state.memory[a] = state.A & state.X

def DCP() -> [(0xC3, indirect_x_address), (0xC7, zeropage_address), (0xCF, absolute_address),
                      (0xD3, indirect_y_address), (0xD7, zeropage_x_address), (0xDB, absolute_y_address),
                      (0xDF, absolute_x_address)]:
    DEC_zpg(state, a)
    CMP(state, state.memory[a])

def ISB() -> [(0xE3, indirect_x_address), (0xE7, zeropage_address), (0xEF, absolute_address),
                      (0xF3, indirect_y_address), (0xF7, zeropage_x_address), (0xFB, absolute_y_address),
                      (0xFF, absolute_x_address)]:
    INC(state, a)
    SBC(state, state.memory[a])

def SLO() -> [(0x03, indirect_x_address), (0x07, zeropage_address), (0x0F, absolute_address),
                      (0x13, indirect_y_address), (0x17, zeropage_x_address), (0x1B, absolute_y_address),
                      (0x1F, absolute_x_address)]:
    ASL_zpg(state, a)
    ORA(state, state.memory[a])

def RLA() -> [(0x23, indirect_x_address), (0x27, zeropage_address), (0x2F, absolute_address),
                      (0x33, indirect_y_address), (0x37, zeropage_x_address), (0x3B, absolute_y_address),
                      (0x3F, absolute_x_address)]:
    ROL_zpg(state, a)
    AND(state, state.memory[a])

def SRE() -> [(0x43, indirect_x_address), (0x47, zeropage_address), (0x4F, absolute_address),
                      (0x53, indirect_y_address), (0x57, zeropage_x_address), (0x5B, absolute_y_address),
                      (0x5F, absolute_x_address)]:
    LSR_zpg(state, a)
    EOR(state, state.memory[a])

def RRA() -> [(0x63, indirect_x_address)]:
    assert False, False
    ROR_zpg(state, a)
    ADC(state, state.memory[a])

#-

def account() -> [(0x02, implied), (0x12, implied), (0x22, implied), (0x32, implied), (0x42, implied),
                  (0x52, implied), (0x62, implied), (0x67, implied), (0x6F, implied), (0x72, implied),
                  (0x73, implied), (0x77, implied), (0x7B, implied), (0x82, implied), (0x0B, implied),
                  (0x2B, implied), (0x4B, implied), (0x6B, implied), (0x7F, implied), (0x89, implied),
                  (0x8B, implied), (0x92, implied), (0x93, implied), (0x9B, implied), (0x9C, implied),
                  (0x9E, implied), (0x9F, implied), (0xAB, implied), (0xB2, implied), (0xBB, implied),
                  (0xC2, implied), (0xD2, implied), (0xCB, implied), (0xE2, implied), (0xF2, implied)]:
    assert None, None

#-


instructions = {}
all_ = [0]*256
for name, func in locals().copy().items():
    if settings := hasattr(func, '__annotations__') and func.__annotations__.get('return'):
        # for opcode, addressing in settings:
        for tup in settings:
            opcode, addressing, wire = tup if len(tup) == 3 else tup + (None,)
            instructions[opcode] = (func, byte_counts[addressing], addressing, wire)
            all_[opcode] = all_[opcode] + 1

for index, count in enumerate(all_):
    if count != 1:
        print(f'0x{index:02X}', count)
        assert False, False

if __name__ == '__main__':
    operations = []
    operation_info = []

    operations_begin = """__device__\nconst RegionComposition instructions[] = {\n    """
    operations_end = "\n};"
    operation_info_begin = """OperationInformation operation_info[] = {\n    """
    operation_info_end = """\n};"""

    format_types = {
        immediate: "Immediate",
        absolute_address: "Absolute",
        zeropage_address: "Zeropage",
        zeropage_dereference: "ZeropageDereference",
        implied: "Implied",
        relative_address: "Address_Relative"
    }

    for opcode in range(256):
        if opcode in [0x4C, 0xA2, 0x86, 0x20, 0xEA, 0x38, 0xB0, 0x18, 0x90, 0xA9, 0xF0,
                      0xD0, 0x85, 0x24, 0x70, 0x50, 0x10, 0x60, 0x78, 0xF8]:
            operation, byte_count, addressing, wire = instructions[opcode]
            region = operation()
            wire = wire or addressing()

            pc_increment = byte_count
            if opcode in [0x4C, 0x20, 0x60]:
                # instructions that directly modify the program counter
                pc_increment = 0

            name = operation.__name__
            if opcode == 0x24:
                name = "BIT"

            operations.append(f'/*0x{opcode:02X}*//*{name}*/{region.struct(wire_region=wire, byte_count=pc_increment)}')
            operation_info.append(f'/*0x{opcode:02X}*//*{name}*/OperationInformation{{.name="{name}", .byte_count={byte_count}, .format_type="{format_types[addressing]}"}}')

        else:
            operations.append(f'/*0x{opcode:02X}*//*NOP*/RegionComposition{{}}')
            operation_info.append(f'/*0x{opcode:02X}*//*NOP*/OperationInformation{{}}')

    operations_text = operations_begin + ',\n    '.join(operations) + operations_end
    operation_info_text = operation_info_begin + ',\n    '.join(operation_info) + operation_info_end

    with open('_instructions.h', 'w') as file:
        file.write('\n' + operations_text + '\n\n' + operation_info_text + '\n')