
import numpy as np


def byte(data):
    return data & 0x00FF


#-

PPU_CONTROL_1 = 0x2000
PPU_CONTROL_2 = 0x2001
PPU_STATUS = 0x2002

PPU_SPR_ADDRESS = 0x2003
PPU_SPR_DATA = 0x2004
PPU_SPR_DMA = 0x4014

PPU_ADDRESS = 0x2006
PPU_DATA = 0x2007
RESET_VECTOR_L = 0xFFFC
RESET_VECTOR_H = 0xFFFD
STACK_ZERO = 0x0100
STACK_OFFSET_INITIAL = 0xFD

"""
PPU_ADDRESS_MAP
0x0000-0x1FFF   PATTERN_MEMORY
0x2000-0x3EFF   NAME_TABLE_MEMORY
0x3F00-0x3FFF   PALETTE_MEMORY
"""

class Behaviors:
    def reset_vblank_on_read(state, address):
        """
        http://wiki.nesdev.com/w/index.php/PPU_registers#PPUSTATUS
        "Reading the status register will clear bit 7 mentioned above
        and also the address latch used by PPUSCROLL and PPUADDR.
        It does not clear the sprite 0 hit or overflow bit."
        """
        if address == PPU_STATUS:
            state.memory.array[PPU_STATUS] &= 0x7F
            state.ppu_address_latch = 0


class State:

    def __init__(self, program_data):
        self.program_counter = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.stack_offset = STACK_OFFSET_INITIAL
        self.ppu_address_latch = 0

        self.memory = Memory(self, program_data)

        self.status_register = {
            'Negative': 0,
            'Zero': 0,
            'Decimal': 0,
            'Interrupt': 0,
            'Carry': 0,
            'Overflow': 0
        }

        self.reset()

    def reset(self):
        # self.program_counter = self.memory[RESET_VECTOR_H]*0x100 + self.memory[RESET_VECTOR_L]
        self.program_counter = 0xC000
        self.stack_offset = STACK_OFFSET_INITIAL


class Memory:
    def __init__(self, state, program_data):
        self.state = state

        self.array = np.zeros(0x10000, dtype=np.uint8)
        # self.array[0x8000:0x8000+len(program_data)] = program_data
        self.array[0xC000:0xC000+len(program_data)] = program_data

    def __getitem__(self, address):
        result = self.array[address]

        Behaviors.reset_vblank_on_read(self.state, address)

        return result

    def __setitem__(self, address, value):
        self.array[address] = value


#-


_ = lambda x: x

def imm(state, data):
    return data

def imm2(state, data1, data2):
    return data2*0x0100 + data1

def zpg(state, data):
    return state.memory[data]

def indx(state, data):
    address = data + state.X
    # return state.memory[(address + 1)*0x0100 + address*data]
    return state.memory[(address + 1)*0x0100 + address]

def zpgx(state, data):
    return (state.memory[data + state.X], data + state.X)

def indy(state, data):
    address = state.memory[data+1]*0x0100 + state.memory[data]
    return address + state.Y

def absy(state, data1, data2):
    address = data2*0x0100 + data1
    return state.memory[address + state.Y]

def abs_read(state, data1, data2):
    address = data2*0x0100 + data1
    result = state.memory[address]

    return result

def abs_x_read(state, data1, data2):
    x = state.X
    # print(f'{data2*256 + data1:04x} + {x:02x} -> {state["Memory"][data2*0x0100 + data1 + x]}')

    return state.memory[data2*0x0100 + data1 + x]


#-


def SEI(state, a) -> [(0x78, _)]: state.status_register['Interrupt'] = 1
def CLD(state, a) -> [(0xD8, _)]: state.status_register['Decimal'] = 0
def SEC(state, a) -> [(0x38, _)]: state.status_register['Carry'] = 1
def NOP(state, a) -> [(0xEA, _)]: pass

def Z_set(state, value):
    state.status_register['Zero'] = 1*(value == 0)

def N_set(state, value):
    state.status_register['Negative'] = 1*(value & 0x80 == 0x80)

def LDA(state, a) -> [(0xA5, zpg), (0xA9, imm), (0xAD, abs_read), (0xBD, abs_x_read), (0xB1, indy)]:
    state.A = a; Z_set(state, a); N_set(state, a)
def LDX(state, a) -> [(0xA2, imm), (0xAE, abs_read)]: state.X = a; Z_set(state, a); N_set(state, a)
def LDY(state, a) -> [(0xA0, imm), (0xAC, abs_read)]: state.Y = a; Z_set(state, a); N_set(state, a)
def DEX(state, a) -> [(0xCA, _)]: state.X = byte(state.X - 1); Z_set(state, state.X); N_set(state, state.X)
def DEY(state, a) -> [(0x88, _)]: state.Y = byte(state.Y - 1); Z_set(state, state.Y); N_set(state, state.Y)
def INY(state, a) -> [(0xC8, _)]: state.Y = byte(state.Y + 1); Z_set(state, state.Y); N_set(state, state.Y)
def INC(state, a) -> [(0xEE, imm2)]: state.memory[a] += 1; Z_set(state, state.memory[a]); N_set(state, state.memory[a])

def BIT(state, a) -> [(0x2C, abs_read)]:
    result = byte(state.A & a)
    Z_set(state, result)
    N_set(state, a)
    state.status_register['Overflow'] = (a >> 6) & 0x01

def CMP(state, a) -> [(0xC9, imm)]:
    result = byte(state.A - a)
    Z_set(state, result); N_set(state, result)
    if a <= state.A:
        state.status_register['Carry'] = 1
    else:
        state.status_register['Carry'] = 0

def CPX(state, a) -> [(0xE0, imm)]:
    result = state.X - a
    Z_set(state, result); N_set(state, result)
    if a <= state.X:
        state.status_register['Carry'] = 1
    else:
        state.status_register['Carry'] = 0

def CPY(state, a) -> [(0xC0, imm)]:
    result = byte(state.Y - a)
    Z_set(state, result); N_set(state, result)
    if a <= state.Y:
        state.status_register['Carry'] = 1
    else:
        state.status_register['Carry'] = 0

def ORA(state, a) -> [(0x09, imm)]: state.A |= a; Z_set(state, state.A); N_set(state, state.A)
def EOR(state, a): state.A ^= a; Z_set(state, state.A); N_set(state, state.A)
def AND(state, a) -> [(0x29, imm)]: state.A &= a; Z_set(state, state.A); N_set(state, state.A)
def ASL(state, a) -> [(0x0A, _)]:
    result = state.A << 1

    Z_set(state, byte(result))
    N_set(state, byte(result))
    state.status_register['Carry'] = 1*(result & 0xFF00 > 0)

    state.A = byte(result)

def LSR(state, a) -> [(0x4A, _)]:
    result = state.A >> 1

    Z_set(state, result)
    N_set(state, result)
    state.status_register['Carry'] = state.A & 0x01

    state.A = result

def ROL(state, a) -> [(0x36, zpgx)]:
    """
    C <- MEM <- C,   N Z C
    """

    (value, memory_location) = a

    result = (value << 1) & state.status_register['Carry']
    state.status_register['Carry'] = 1*(result & 0xFF00 > 0)
    result = byte(result)
    Z_set(state, result)
    N_set(state, result)

    state.memory[memory_location] = result


once = set()
def ADC(state, a) -> [(0x65, zpg)]:
    """
    A + M + C -> A, C                N Z C i d V

    The overflow flag is set
    when the sign or bit 7 is changed due to the result exceeding +127
    or -128, otherwise overflow is reset.
    """

    result = state.A + a + state.status_register['Carry']

    Z_set(state, byte(result))
    N_set(state, byte(result))
    state.status_register['Carry'] = 1*(result & 0xFF00 > 0)

    v1 =  ~(state.A ^ a) & (state.A ^ result) & 0x0080
    flag_0 = state.A >> 7
    flag_1 = result >> 7
    v2 = (flag_1 != flag_0) and state.status_register['Carry']

    state.status_register['Overflow'] = v1
    if v1 != 1*v2:
        if (v1, v2) not in once:
            once.add((v1, v2))
            print("V", v1, 1*v2)

    state.A = byte(result)



def STA(state, a) -> [(0x8D, imm2), (0x85, imm), (0x91, indy), (0x99, absy)]:
    state.memory[a] = state.A
def STX(state, a) -> [(0x86, imm)]: state.memory[a] = state.X
def TXA(state, a) -> [(0x8A, _)]: state.A = state.X; Z_set(state, state.A); N_set(state, state.A)
def TYA(state, a) -> [(0x98, _)]: state.A = state.Y; Z_set(state, state.A); N_set(state, state.A)
def TAX(state, a) -> [(0xAA, _)]: state.X = state.A; Z_set(state, state.X); N_set(state, state.X)

def TXS(state, a) -> [(0x9A, _)]: state.stack_offset = state.X

def BPL(state, a) -> [(0x10, imm)]:
    if state.status_register['Negative'] == 0:
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

def BMI(state, a) -> [(0x30, imm)]:
    if state.status_register['Negative'] == 1:
        state.program_counter += np.int8(a)

def JMP(state, a) -> [(0x4C, imm2)]:
    state.program_counter = a

def JSR(state, a) -> [(0x20, imm2)]:
    # Stack is from range 0x0100-0x1FF and grows down from 0x0100 + 0xFD.
    pc_H = state.program_counter >> 8
    pc_L = state.program_counter & 0x00FF

    state.memory[STACK_ZERO + state.stack_offset] = pc_H
    state.stack_offset -= 1
    state.memory[STACK_ZERO + state.stack_offset] = pc_L
    state.stack_offset -= 1

    state.program_counter = a

def NMI(state):
    nmi_L = state.memory[0xFFFA]
    nmi_H = state.memory[0xFFFB]
    data = nmi_H*0x0100 + nmi_L

    JSR(state, data)

def RTS(state, a) -> [(0x60, _)]:
    state.stack_offset += 1
    pc_L = state.memory[STACK_ZERO + state.stack_offset]
    state.stack_offset += 1
    pc_H = state.memory[STACK_ZERO + state.stack_offset]

    state.program_counter = (pc_H << 8) + pc_L

def PHA(state, a) -> [(0x48, _)]:
    state.memory[STACK_ZERO + state.stack_offset] = state.A
    state.stack_offset -= 1

def PLA(state, a) -> [(0x68, _)]:
    state.stack_offset += 1
    state.A = state.memory[STACK_ZERO + state.stack_offset]

    Z_set(state, state.A)
    N_set(state, state.A)


#-


instructions = {}
byte_counts = {_: 1, imm: 2, imm2: 3, zpg: 2, zpgx: 2, abs_read: 3, abs_x_read: 3, indy: 2, absy: 3}
for name, func in locals().copy().items():
    if settings := hasattr(func, '__annotations__') and func.__annotations__.get('return'):
        for opcode, addressing in settings:
            instructions[opcode] = (func, byte_counts[addressing], addressing)
