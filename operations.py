
import numpy as np


def byte(data):
    return data & 0x00FF


#-


_ = lambda x: x

def imm(state, data):
    return data

def imm2(state, data1, data2):
    return data2*0x0100 + data1

def indx(state, data):
    address = data + state['X']
    return state['Memory'][(address + 1)*0x0100 + address*data]

def indy(state, data):
    address = state['Memory'][data+1]*0x0100 + state['Memory'][data]
    return address + state['Y']

def absy(state, data1, data2):
    address = data2*0x0100 + data1
    return state['Memory'][address + state['Y']]

def abs_read(state, data1, data2):
    print(f'{data2*256 + data1:04x} -> {state["Memory"][data2*0x0100 + data1]}')

    return state['Memory'][data2*0x0100 + data1]

def abs_x_read(state, data1, data2):
    x = state['X']
    print(f'{data2*256 + data1:04x} + {x:02x} -> {state["Memory"][data2*0x0100 + data1 + x]}')

    return state['Memory'][data2*0x0100 + data1 + x]


#-


def SEI(state, a) -> [(0x78, _)]: state['StatusRegister']['Interrupt'] = 1
def CLD(state, a) -> [(0xD8, _)]: state['StatusRegister']['Decimal'] = 0

def Z_set(state, value):
    state['StatusRegister']['Zero'] = 1*(value == 0)

def N_set(state, value):
    state['StatusRegister']['Negative'] = 1*(value & 0x80 == 0x80)

def LDA(state, a) -> [(0xA9, imm), (0xAD, abs_read), (0xBD, abs_x_read)]:
    state['Accumulator'] = a; Z_set(state, a); N_set(state, a)
def LDX(state, a) -> [(0xA2, imm)]: state['X'] = a; Z_set(state, a); N_set(state, a)
def LDY(state, a) -> [(0xA0, imm)]: state['Y'] = a; Z_set(state, a); N_set(state, a)
def DEX(state, a) -> [(0xCA, _)]: state['X'] = byte(state['X'] - 1); Z_set(state, state['X']); N_set(state, state['X'])
def DEY(state, a) -> [(0x88, _)]: state['Y'] = byte(state['Y'] - 1); Z_set(state, state['Y']); N_set(state, state['Y'])
def INY(state, a) -> [(0xC8, _)]: state['Y'] = byte(state['Y'] + 1); Z_set(state, state['Y']); N_set(state, state['Y'])
def INC(state, a) -> [(0xEE, imm2)]: state['Memory'][a] += 1; Z_set(state, state['Memory'][a]); N_set(state, state['Memory'][a])

def BIT(state, a) -> [(0x2C, abs_read)]:
    result = byte(state['Accumulator'] & a)
    Z_set(state, result)
    N_set(state, a)
    state['StatusRegister']['Overflow'] = (a >> 6) & 0x01

def CMP(state, a) -> [(0xC9, imm)]:
    result = state['Accumulator'] - a
    Z_set(state, result); N_set(state, result)
    if a <= state['Accumulator']:
        state['StatusRegister']['Carry'] = 1
    else:
        state['StatusRegister']['Carry'] = 0

def CPX(state, a) -> [(0xE0, imm)]:
    result = state['X'] - a
    Z_set(state, result); N_set(state, result)
    if a <= state['X']:
        state['StatusRegister']['Carry'] = 1
    else:
        state['StatusRegister']['Carry'] = 0

def CPY(state, a) -> [(0xC0, imm)]:
    result = byte(state['Y'] - a)
    Z_set(state, result); N_set(state, result)
    if a <= state['Y']:
        state['StatusRegister']['Carry'] = 1
    else:
        state['StatusRegister']['Carry'] = 0

def ORA(state, a) -> [(0x09, imm)]: state['Accumulator'] |= a; Z_set(state, state['Accumulator']); N_set(state, state['Accumulator'])
def EOR(state, a): state['Accumulator'] ^= a; Z_set(state, state['Accumulator']); N_set(state, state['Accumulator'])
def AND(state, a) -> [(0x29, imm)]: state['Accumulator'] &= a; Z_set(state, state['Accumulator']); N_set(state, state['Accumulator'])

def STA(state, a) -> [(0x8D, imm2), (0x85, imm), (0x91, indy), (0x99, absy)]:
    state['Memory'][a] = state['Accumulator']
def STX(state, a) -> [(0x86, imm)]: state['Memory'][a] = state['X']
def TXA(state, a) -> [(0x8A, _)]: state['Accumulator'] = state['X']; Z_set(state, state['Accumulator']); N_set(state, state['Accumulator'])

def TXS(state, a) -> [(0x9A, _)]: state['StackRegister'] = state['X']

def BPL(state, a) -> [(0x10, imm)]:
    if state['StatusRegister']['Negative'] == 0:
        state['ProgramCounter'] += np.int8(a)

def BCS(state, a) -> [(0xB0, imm)]:
    if state['StatusRegister']['Carry'] == 1:
        state['ProgramCounter'] += np.int8(a)

def BNE(state, a) -> [(0xD0, imm)]:
    if state['StatusRegister']['Zero'] == 0:
        state['ProgramCounter'] += np.int8(a)

def JMP(state, a) -> [(0x4C, imm2)]:
    state['ProgramCounter'] = a

def JSR(state, a) -> [(0x20, imm2)]:
    # Stack is from range 0x0100-0x1FF and grows down from 0x0100 + 0xFD.
    pc_H = state['ProgramCounter'] >> 8
    pc_L = state['ProgramCounter'] & 0x00FF

    state['Memory'][state['StackZero'] + state['StackOffset']] = pc_H
    state['StackOffset'] -= 1
    state['Memory'][state['StackZero'] + state['StackOffset']] = pc_L
    state['StackOffset'] -= 1

    state['ProgramCounter'] = a

def RTS(state, a) -> [(0x60, _)]:
    state['StackOffset'] += 1
    pc_L = state['Memory'][state['StackZero'] + state['StackOffset']]
    state['StackOffset'] += 1
    pc_H = state['Memory'][state['StackZero'] + state['StackOffset']]

    state['ProgramCounter'] = (pc_H << 8) + pc_L


#-


instructions = {}
byte_counts = {_: 1, imm: 2, imm2: 3, abs_read: 3, abs_x_read: 3, indy: 2, absy: 3}
for name, func in locals().copy().items():
    if settings := callable(func) and func.__annotations__.get('return'):
        for opcode, addressing in settings:
            instructions[opcode] = (func, byte_counts[addressing], addressing)
