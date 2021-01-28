
import numpy as np
from behaviors import Behaviors

#-

def byte(data):
    return data & 0x00FF

#-

STACK_ZERO = 0x0100

#-

def _(state, _data1, _data2):
    pass

def imm(state, data1, _data2):
    return data1

def imm2(state, data1, data2):
    return data2*0x0100 + data1

def zpg(state, data1, _data2):
    return state.memory[data1]

def indx(state, data1, _data2):
    L = state.memory[byte(data1 + state.X)]
    H = state.memory[byte(data1 + state.X + 1)]
    address = H*0x0100 + L
    return state.memory[address]

def indx_write(state, data1, _data2):
    L = state.memory[byte(data1 + state.X)]
    H = state.memory[byte(data1 + state.X + 1)]
    return H*0x0100 + L

def zpgx_read(state, data1, _data2):
    return state.memory[data1 + state.X]

def zpgx_write(state, data1, _data2):
    return data1 + state.X

def zpgy_read(state, data1, _data2):
    return state.memory[data1 + state.Y]

def zpgy_write(state, data1, _data2):
    return data1 + state.Y

def zpg_write(state, data1, _data2):
    return data1

def abs_write(state, data1, data2):
    return data2*0x0100 + data1

def indy(state, data1, _data2):
    # check
    address = state.memory[data1+1]*0x0100 + state.memory[data1]
    return address + state.Y

def indy_read(state, data1, _data2):
    address = state.memory[byte(data1+1)]*0x0100 + state.memory[data1]
    return state.memory[address + state.Y]

def ind(state, data1, data2):
    # indirect wraps at page boundary
    L = state.memory[data2*0x0100 + data1]
    H = state.memory[data2*0x0100 + byte(data1 + 1)]

    return H*0x0100 + L

def absy(state, data1, data2):
    address = data2*0x0100 + data1
    return state.memory[address + state.Y]

def absy_write(state, data1, data2):
    address = data2*0x0100 + data1
    return address + state.Y

def abs_read(state, data1, data2):
    address = data2*0x0100 + data1
    result = state.memory[address]

    return result

def absx_read(state, data1, data2):
    return state.memory[data2*0x0100 + data1 + state.X]

def absx_write(state, data1, data2):
    return data2*0x0100 + data1 + state.X


#-


def SEI(state, a) -> [(0x78, _)]: state.status_register['Interrupt'] = 1
def CLD(state, a) -> [(0xD8, _)]: state.status_register['Decimal'] = 0
def SEC(state, a) -> [(0x38, _)]: state.status_register['Carry'] = 1
def CLC(state, a) -> [(0x18, _)]: state.status_register['Carry'] = 0
def SED(state, a) -> [(0xF8, _)]: state.status_register['Decimal'] = 1
def CLV(state, a) -> [(0xB8, _)]: state.status_register['Overflow'] = 0
def NOP(state, a) -> [(0x1A, _), (0x3A, _), (0x5A, _),
                      (0x7A, _), (0xDA, _), (0xFA, _),

                      (0x04, imm), (0x14, imm), (0x34, imm),
                      (0x44, imm), (0x54, imm), (0x64, imm),
                      (0x80, imm),
                      (0x74, imm), (0xD4, imm), (0xF4, imm),

                      (0x0C, imm2), (0x1C, imm2), (0x3C, imm2),
                      (0x5C, imm2), (0x7C, imm2), (0xDC, imm2),
                      (0xFC, imm2),

                      (0xEA, _)]: pass

def Z_set(state, value):
    state.status_register['Zero'] = 1*(value == 0)

def N_set(state, value):
    state.status_register['Negative'] = 1*(value & 0x80 == 0x80)

def LDA(state, a) -> [(0xA1, indx), (0xA5, zpg), (0xA9, imm), (0xAD, abs_read), (0xB1, indy_read), (0xB5, zpgx_read), (0xB9, absy), (0xBD, absx_read)]:
    state.A = a; Z_set(state, a); N_set(state, a)
def LDX(state, a) -> [(0xA2, imm), (0xA6, zpg), (0xAE, abs_read), (0xB6, zpgy_read), (0xBE, absy)]: state.X = a; Z_set(state, a); N_set(state, a)
def LDY(state, a) -> [(0xA0, imm), (0xA4, zpg), (0xAC, abs_read), (0xB4, zpgx_read), (0xBC, absx_read)]: state.Y = a; Z_set(state, a); N_set(state, a)
def DEX(state, a) -> [(0xCA, _)]: state.X = byte(state.X - 1); Z_set(state, state.X); N_set(state, state.X)
def DEY(state, a) -> [(0x88, _)]: state.Y = byte(state.Y - 1); Z_set(state, state.Y); N_set(state, state.Y)
def INY(state, a) -> [(0xC8, _)]: state.Y = byte(state.Y + 1); Z_set(state, state.Y); N_set(state, state.Y)
def INC(state, a) -> [(0xE6, imm), (0xEE, imm2), (0xF6, zpgx_write), (0xFE, absx_write)]: state.memory[a] += 1; Z_set(state, state.memory[a]); N_set(state, state.memory[a])
def INX(state, a) -> [(0xE8, _)]: state.X = byte(state.X + 1); Z_set(state, state.X); N_set(state, state.X)

def LAX(state, a) -> [(0xA3, indx), (0xA7, zpg), (0xAF, abs_read), (0xB3, indy_read), (0xB7, zpgy_read), (0xBF, absy)]:
    LDA(state, a)
    TAX(state, a)

def SAX(state, a) -> [(0x83, indx_write), (0x87, zpg_write), (0x8F, abs_write), (0x97, zpgy_write)]:
    state.memory[a] = state.A & state.X

def DEC_zpg(state, a) -> [(0xC6, imm), (0xCE, imm2), (0xD6, zpgx_write), (0xDE, absx_write)]:
    state.memory[a] -= 1
    Z_set(state, state.memory[a])
    N_set(state, state.memory[a])

def BIT(state, a) -> [(0x2C, abs_read)]:
    result = byte(state.A & a)
    Z_set(state, result)
    N_set(state, a)
    state.status_register['Overflow'] = (a >> 6) & 0x01

def BIT_zpg(state, a) -> [(0x24, zpg)]:
    state.status_register['Negative'] = (a >> 7) & 0x01
    state.status_register['Overflow'] = (a >> 6) & 0x01
    Z_set(state, byte(state.A & a))

def CMP(state, a) -> [(0xC1, indx), (0xC5, zpg), (0xC9, imm), (0xCD, abs_read), (0xD1, indy_read), (0xD5, zpgx_read), (0xD9, absy), (0xDD, absx_read)]:
    result = byte(state.A - a)
    Z_set(state, result); N_set(state, result)
    if a <= state.A:
        state.status_register['Carry'] = 1
    else:
        state.status_register['Carry'] = 0

def CPX(state, a) -> [(0xE0, imm), (0xE4, zpg), (0xEC, abs_read)]:
    result = state.X - a
    Z_set(state, result); N_set(state, result)
    if a <= state.X:
        state.status_register['Carry'] = 1
    else:
        state.status_register['Carry'] = 0

def CPY(state, a) -> [(0xC0, imm), (0xC4, zpg), (0xCC, abs_read)]:
    result = byte(state.Y - a)
    Z_set(state, result); N_set(state, result)
    if a <= state.Y:
        state.status_register['Carry'] = 1
    else:
        state.status_register['Carry'] = 0

def ORA(state, a) -> [(0x01, indx), (0x05, zpg), (0x09, imm), (0x0D, abs_read), (0x11, indy_read), (0x15, zpgx_read), (0x19, absy), (0x1D, absx_read)]: state.A |= a; Z_set(state, state.A); N_set(state, state.A)
def EOR(state, a) -> [(0x41, indx), (0x45, zpg), (0x49, imm), (0x4D, abs_read), (0x51, indy_read), (0x55, zpgx_read), (0x59, absy), (0x5D, absx_read)]: state.A ^= a; Z_set(state, state.A); N_set(state, state.A)
def AND(state, a) -> [(0x21, indx), (0x25, zpg), (0x29, imm), (0x2D, abs_read), (0x31, indy_read), (0x35, zpgx_read), (0x39, absy), (0x3D, absx_read)]: state.A &= a; Z_set(state, state.A); N_set(state, state.A)
def ASL(state, a) -> [(0x0A, _)]:
    result = state.A << 1

    Z_set(state, byte(result))
    N_set(state, byte(result))
    state.status_register['Carry'] = 1*(result & 0xFF00 > 0)

    state.A = byte(result)

def ASL_zpg(state, a) -> [(0x06, imm), (0x0E, imm2), (0x16, zpgx_write), (0x1E, absx_write)]:
    result = byte(state.memory[a] << 1)

    Z_set(state, result)
    N_set(state, result)
    state.status_register['Carry'] = state.memory[a] >> 7

    state.memory[a] = result

def LSR(state, a) -> [(0x4A, _)]:
    result = state.A >> 1

    Z_set(state, result)
    N_set(state, result)
    state.status_register['Carry'] = state.A & 0x01

    state.A = result

def LSR_zpg(state, a) -> [(0x46, imm), (0x4E, imm2), (0x56, zpgx_write), (0x5E, absx_write)]:
    result = state.memory[a] >> 1

    Z_set(state, result)
    N_set(state, result)
    state.status_register['Carry'] = state.memory[a] & 0x01

    state.memory[a] = result

def ROL(state, a) -> [(0x2A, _)]:
    result = (state.A << 1) | state.status_register['Carry']
    state.status_register['Carry'] = 1*(result & 0xFF00 > 0)
    result = byte(result)
    Z_set(state, result)
    N_set(state, result)

    state.A = result

def ROL_zpg(state, a) -> [(0x26, imm), (0x2E, imm2), (0x36, zpgx_write), (0x3E, absx_write)]:
    result = (state.memory[a] << 1) | state.status_register['Carry']
    state.status_register['Carry'] = 1*(result & 0xFF00 > 0)
    result = byte(result)

    Z_set(state, result)
    N_set(state, result)

    state.memory[a] = result

def ROR(state, a) -> [(0x6A, _)]:
    result = state.status_register['Carry'] << 7
    state.status_register['Carry'] = state.A & 0x01
    result += state.A >> 1

    N_set(state, result)
    Z_set(state, result)

    state.A = result

def ROR_zpg(state, a) -> [(0x66, imm), (0x6E, imm2), (0x76, zpgx_write), (0x7E, absx_write)]:
    result = state.status_register['Carry'] << 7
    state.status_register['Carry'] = state.memory[a] & 0x01
    result += state.memory[a] >> 1

    N_set(state, result)
    Z_set(state, result)

    state.memory[a] = result

once = set()
def ADC(state, a) -> [(0x61, indx), (0x65, zpg), (0x69, imm), (0x6D, abs_read), (0x71, indy_read), (0x75, zpgx_read), (0x79, absy), (0x7D, absx_read)]:
    result = state.A + a + state.status_register['Carry']

    Z_set(state, byte(result))
    N_set(state, byte(result))
    state.status_register['Carry'] = 1*(result & 0xFF00 > 0)

    v1 =  ~(state.A ^ a) & (state.A ^ result) & 0x0080
    flag_0 = state.A >> 7
    flag_1 = result >> 7
    v2 = (flag_1 != flag_0) and state.status_register['Carry']

    state.status_register['Overflow'] = 1*(v1 > 0)
    if v1 != 1*v2:
        if (v1, v2) not in once:
            once.add((v1, v2))
            print("V", v1, 1*v2)

    state.A = byte(result)

def SBC(state, a) -> [(0xE1, indx), (0xE5, zpg), (0xE9, imm), (0xEB, imm), (0xED, abs_read),
                      (0xF1, indy_read), (0xF5, zpgx_read), (0xF9, absy), (0xFD, absx_read)]:
    a ^= 0xFF
    result = state.A + a + state.status_register['Carry']

    Z_set(state, byte(result))
    N_set(state, byte(result))
    state.status_register['Carry'] = 1*(result & 0xFF00 > 0)

    v1 =  ~(state.A ^ a) & (state.A ^ result) & 0x0080
    state.status_register['Overflow'] = 1*(v1 > 0)

    state.A = byte(result)

def DCP(state, a) -> [(0xC3, indx_write), (0xC7, zpg_write), (0xCF, abs_write), (0xD3, indy), (0xD7, zpgx_write),
                      (0xDB, absy_write), (0xDF, absx_write)]:
    DEC_zpg(state, a)
    CMP(state, state.memory[a])

def ISB(state, a) -> [(0xE3, indx_write), (0xE7, zpg_write), (0xEF, abs_write), (0xF3, indy), (0xF7, zpgx_write),
                      (0xFB, absy_write), (0xFF, absx_write)]:
    INC(state, a)
    SBC(state, state.memory[a])

def SLO(state, a) -> [(0x03, indx_write), (0x07, zpg_write), (0x0F, abs_write), (0x13, indy), (0x17, zpgx_write),
                      (0x1B, absy_write), (0x1F, absx_write)]:
    ASL_zpg(state, a)
    ORA(state, state.memory[a])

def RLA(state, a) -> [(0x23, indx_write), (0x27, zpg_write), (0x2F, abs_write), (0x33, indy), (0x37, zpgx_write),
                      (0x3B, absy_write), (0x3F, absx_write)]:
    ROL_zpg(state, a)
    AND(state, state.memory[a])

def SRE(state, a) -> [(0x43, indx_write), (0x47, zpg_write), (0x4F, abs_write), (0x53, indy), (0x57, zpgx_write),
                      (0x5B, absy_write), (0x5F, absx_write)]:
    LSR_zpg(state, a)
    EOR(state, state.memory[a])

def RRA(state, a) -> [(0x63, indx_write)]:
    assert False, False
    ROR_zpg(state, a)
    ADC(state, state.memory[a])

def STA(state, a) -> [(0x81, indx_write), (0x8D, imm2), (0x85, imm), (0x91, indy), (0x95, zpgx_write), (0x99, absy_write), (0x9D, absx_write)]:
    state.memory[a] = state.A
def STX(state, a) -> [(0x86, imm), (0x8E, imm2), (0x96, zpgy_write)]: state.memory[a] = state.X
def STY(state, a) -> [(0x84, imm), (0x8C, imm2), (0x94, zpgx_write)]: state.memory[a] = state.Y
def TXA(state, a) -> [(0x8A, _)]: state.A = state.X; Z_set(state, state.A); N_set(state, state.A)
def TYA(state, a) -> [(0x98, _)]: state.A = state.Y; Z_set(state, state.A); N_set(state, state.A)
def TAX(state, a) -> [(0xAA, _)]: state.X = state.A; Z_set(state, state.X); N_set(state, state.X)
def TAY(state, a) -> [(0xA8, _)]: state.Y = state.A; Z_set(state, state.Y); N_set(state, state.Y)
def TSX(state, a) -> [(0xBA, _)]: state.X = state.stack_offset; Z_set(state, state.X); N_set(state, state.X)

def TXS(state, a) -> [(0x9A, _)]: state.stack_offset = state.X

def BPL(state, a) -> [(0x10, imm)]:
    if state.status_register['Negative'] == 0:
        state.program_counter += np.int8(a)

def BMI(state, a) -> [(0x30, imm)]:
    if state.status_register['Negative'] == 1:
        state.program_counter += np.int8(a)

def BCC(state, a) -> [(0x90, imm)]:
    if state.status_register['Carry'] == 0:
        state.program_counter += np.int8(a)

def BCS(state, a) -> [(0xB0, imm)]:
    if state.status_register['Carry'] == 1:
        state.program_counter += np.int8(a)

def BNE(state, a) -> [(0xD0, imm)]:
    if state.status_register['Zero'] == 0:
        state.program_counter += np.int8(a)

def BEQ(state, a) -> [(0xF0, imm)]:
    if state.status_register['Zero'] == 1:
        state.program_counter += np.int8(a)

def BVC(state, a) -> [(0x50, imm)]:
    if state.status_register['Overflow'] == 0:
        state.program_counter += np.int8(a)

def JMP(state, a) -> [(0x4C, imm2), (0x6C, ind)]:
    state.program_counter = a

def BVS(state, a) -> [(0x70, imm)]:
    if state.status_register['Overflow'] == 1:
        state.program_counter += np.int8(a)

def JSR(state, a) -> [(0x20, imm2)]:
    # Stack is from range 0x0100-0x1FF and grows down from 0x0100 + 0xFD.
    # Adjust for current program counter incrementing.
    program_counter = state.program_counter - 1
    pc_H = program_counter >> 8
    pc_L = program_counter & 0x00FF

    state.memory[STACK_ZERO + state.stack_offset] = pc_H
    state.stack_offset -= 1
    state.memory[STACK_ZERO + state.stack_offset] = pc_L
    state.stack_offset -= 1

    state.program_counter = a

def BRK(state, a) -> [(0x00, _)]:
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

def NMI(state):
    nmi_L = state.memory[0xFFFA]
    nmi_H = state.memory[0xFFFB]
    data = nmi_H*0x0100 + nmi_L

    JSR(state, data)

def RTI(state, a) -> [(0x40, _)]:
    state.stack_offset += 1
    status_register = state.memory[STACK_ZERO + state.stack_offset]
    status_register = Behaviors.read_special_status_bits_on_pull(state, status_register)
    state.status_register_byte_set(status_register)

    state.stack_offset += 1
    pc_L = state.memory[STACK_ZERO + state.stack_offset]
    state.stack_offset += 1
    pc_H = state.memory[STACK_ZERO + state.stack_offset]

    program_counter = (pc_H <<8) + pc_L
    state.program_counter = program_counter

def RTS(state, a) -> [(0x60, _)]:
    state.stack_offset += 1
    pc_L = state.memory[STACK_ZERO + state.stack_offset]
    state.stack_offset += 1
    pc_H = state.memory[STACK_ZERO + state.stack_offset]

    program_counter = (pc_H << 8) + pc_L

    # Adjust for current program counter incrementing.
    state.program_counter = program_counter + 1

def PHA(state, a) -> [(0x48, _)]:
    state.memory[STACK_ZERO + state.stack_offset] = state.A
    state.stack_offset -= 1

def PHP(state, a) -> [(0x08, _)]:
    status_register = state.status_register_byte()
    status_register = Behaviors.write_special_status_bits_on_push(PHP, status_register)

    state.memory[STACK_ZERO + state.stack_offset] = status_register
    state.stack_offset -= 1

def PLA(state, a) -> [(0x68, _)]:
    state.stack_offset += 1
    state.A = state.memory[STACK_ZERO + state.stack_offset]

    Z_set(state, state.A)
    N_set(state, state.A)

def PLP(state, a) -> [(0x28, _)]:
    state.stack_offset += 1
    data = state.memory[STACK_ZERO + state.stack_offset]
    data = Behaviors.read_special_status_bits_on_pull(state, data)

    state.status_register_byte_set(data)


#-


instructions = {}
byte_counts = {_: 1, imm: 2, imm2: 3, zpg: 2, zpg_write: 2, zpgx_read: 2, zpgx_write: 2, zpgy_read: 2, zpgy_write: 2, abs_read: 3,
               abs_write: 3, absx_read: 3, absx_write: 3, indx: 2, indx_write: 2, indy: 2, indy_read: 2, ind: 3, absy: 3, absy_write: 3}
for name, func in locals().copy().items():
    if settings := hasattr(func, '__annotations__') and func.__annotations__.get('return'):
        for opcode, addressing in settings:
            instructions[opcode] = (func, byte_counts[addressing], addressing)
