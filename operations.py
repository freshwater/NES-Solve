
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

    def write_special_status_bits_on_push(function, status_register):
        """
        https://wiki.nesdev.com/w/index.php/Status_flags#The_B_flag
        Two interrupts (/IRQ and /NMI) and two instructions (PHP and BRK)
        push the flags to the stack. In the byte pushed, bit 5 is always
        set to 1, and bit 4 is 1 if from an instruction (PHP or BRK) or 0
        if from an interrupt line being pulled low (/IRQ or /NMI). This is
        the only time and place where the B flag actually exists: not in
        the status register itself, but in bit 4 of the copy that is
        written to the stack.
        """
        return status_register | 0x20 | (0x10 if function in [PHP] else 0x00)

    def read_special_status_bits_on_pull(state, data):
        """
        https://wiki.nesdev.com/w/index.php/Status_flags#The_B_flag
        Two instructions (PLP and RTI) pull a byte from the stack and set all
        the flags. They ignore bits 5 and 4.
        """
        bits = state.status_register_byte() & 0x30
        data &= (0xFF - 0x30)
        data |= bits

        return data


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
            'Overflow': 0,
            'Unused': 1,
            'Break': 0,
            'Decimal': 0,
            'Interrupt': 1,
            'Zero': 0,
            'Carry': 0,
        }

        self.reset()

    def reset(self):
        # self.program_counter = self.memory[RESET_VECTOR_H]*0x100 + self.memory[RESET_VECTOR_L]
        self.program_counter = 0xC000
        self.stack_offset = STACK_OFFSET_INITIAL

    def status_register_byte(self):
        sr = self.status_register
        status_register = (
            (sr['Negative'] << 7) + (sr['Overflow'] << 6) + (sr['Unused'] << 5) + (sr['Break'] << 4) +
            (sr['Decimal'] << 3) + (sr['Interrupt'] << 2) + (sr['Zero'] << 1) + (sr['Carry'] << 0))

        return status_register

    def status_register_byte_set(self, sr):
        self.status_register = {
            key: (sr >> bit) & 0x01
            for key, bit in zip(self.status_register.keys(), [7, 6, 5, 4, 3, 2, 1, 0])}


class Memory:
    def __init__(self, state, program_data):
        self.state = state

        self.array = np.zeros(0x10000, dtype=np.uint8)
        # self.array[0x8000:0x8000+len(program_data)] = program_data
        self.array[0xC000:0xC000+len(program_data)] = program_data

    def __getitem__(self, address):
        result = self.array[address & 0xFFFF]

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
    L = state.memory[byte(data + state.X)]
    H = state.memory[byte(data + state.X + 1)]
    address = H*0x0100 + L
    return state.memory[address]

def indx_write(state, data):
    L = state.memory[byte(data + state.X)]
    H = state.memory[byte(data + state.X + 1)]
    return H*0x0100 + L

def zpgx(state, data):
    return (state.memory[data + state.X], data + state.X)

def indy(state, data):
    # check
    address = state.memory[data+1]*0x0100 + state.memory[data]
    return address + state.Y

def indy_read(state, data):
    address = state.memory[byte(data+1)]*0x0100 + state.memory[data]
    return state.memory[address + state.Y]

def ind(state, data1, data2):
    # indirect wraps at page boundary
    L = state.memory[data2*0x0100 + data1]
    H = state.memory[data2*0x0100 + byte(data1 + 1)]

    return H*0x0100 + L

def absy(state, data1, data2):
    address = data2*0x0100 + data1
    return state.memory[address + state.Y]

def abs_read(state, data1, data2):
    address = data2*0x0100 + data1
    result = state.memory[address]

    return result

def abs_x_read(state, data1, data2):
    x = state.X

    return state.memory[data2*0x0100 + data1 + x]


#-


def SEI(state, a) -> [(0x78, _)]: state.status_register['Interrupt'] = 1
def CLD(state, a) -> [(0xD8, _)]: state.status_register['Decimal'] = 0
def SEC(state, a) -> [(0x38, _)]: state.status_register['Carry'] = 1
def CLC(state, a) -> [(0x18, _)]: state.status_register['Carry'] = 0
def SED(state, a) -> [(0xF8, _)]: state.status_register['Decimal'] = 1
def CLV(state, a) -> [(0xB8, _)]: state.status_register['Overflow'] = 0
def NOP(state, a) -> [(0xEA, _)]: pass

def Z_set(state, value):
    state.status_register['Zero'] = 1*(value == 0)

def N_set(state, value):
    state.status_register['Negative'] = 1*(value & 0x80 == 0x80)

def LDA(state, a) -> [(0xA1, indx), (0xA5, zpg), (0xA9, imm), (0xAD, abs_read), (0xB1, indy_read), (0xBD, abs_x_read)]:
    state.A = a; Z_set(state, a); N_set(state, a)
def LDX(state, a) -> [(0xA2, imm), (0xA6, zpg), (0xAE, abs_read)]: state.X = a; Z_set(state, a); N_set(state, a)
def LDY(state, a) -> [(0xA0, imm), (0xA4, zpg), (0xAC, abs_read)]: state.Y = a; Z_set(state, a); N_set(state, a)
def DEX(state, a) -> [(0xCA, _)]: state.X = byte(state.X - 1); Z_set(state, state.X); N_set(state, state.X)
def DEY(state, a) -> [(0x88, _)]: state.Y = byte(state.Y - 1); Z_set(state, state.Y); N_set(state, state.Y)
def INY(state, a) -> [(0xC8, _)]: state.Y = byte(state.Y + 1); Z_set(state, state.Y); N_set(state, state.Y)
def INC(state, a) -> [(0xE6, imm), (0xEE, imm2)]: state.memory[a] += 1; Z_set(state, state.memory[a]); N_set(state, state.memory[a])
def INX(state, a) -> [(0xE8, _)]: state.X = byte(state.X + 1); Z_set(state, state.X); N_set(state, state.X)

def DEC_zpg(state, a) -> [(0xC6, imm), (0xCE, imm2)]:
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

def CMP(state, a) -> [(0xC1, indx), (0xC5, zpg), (0xC9, imm), (0xCD, abs_read), (0xD1, indy_read)]:
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

def ORA(state, a) -> [(0x01, indx), (0x05, zpg), (0x09, imm), (0x0D, abs_read), (0x11, indy_read)]: state.A |= a; Z_set(state, state.A); N_set(state, state.A)
def EOR(state, a) -> [(0x41, indx), (0x45, zpg), (0x49, imm), (0x4D, abs_read), (0x51, indy_read)]: state.A ^= a; Z_set(state, state.A); N_set(state, state.A)
def AND(state, a) -> [(0x21, indx), (0x25, zpg), (0x29, imm), (0x2D, abs_read), (0x31, indy_read)]: state.A &= a; Z_set(state, state.A); N_set(state, state.A)
def ASL(state, a) -> [(0x0A, _)]:
    result = state.A << 1

    Z_set(state, byte(result))
    N_set(state, byte(result))
    state.status_register['Carry'] = 1*(result & 0xFF00 > 0)

    state.A = byte(result)

def ASL_zpg(state, a) -> [(0x06, imm), (0x0E, imm2)]:
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

def LSR_zpg(state, a) -> [(0x46, imm), (0x4E, imm2)]:
    result = state.memory[a] >> 1

    Z_set(state, result)
    N_set(state, result)
    state.status_register['Carry'] = state.A & 0x01

    state.memory[a] = result

def ROL_zpgx(state, a) -> [(0x36, zpgx)]:
    (value, memory_location) = a

    result = (value << 1) & state.status_register['Carry']
    state.status_register['Carry'] = 1*(result & 0xFF00 > 0)
    result = byte(result)
    Z_set(state, result)
    N_set(state, result)

    state.memory[memory_location] = result

def ROL(state, a) -> [(0x2A, _)]:
    result = (state.A << 1) | state.status_register['Carry']
    state.status_register['Carry'] = 1*(result & 0xFF00 > 0)
    result = byte(result)
    Z_set(state, result)
    N_set(state, result)

    state.A = result

def ROL_zpg(state, a) -> [(0x26, imm), (0x2E, imm2)]:
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

def ROR_zpg(state, a) -> [(0x66, imm), (0x6E, imm2)]:
    result = state.status_register['Carry'] << 7
    state.status_register['Carry'] = state.memory[a] & 0x01
    result += state.memory[a] >> 1

    N_set(state, result)
    Z_set(state, result)

    state.memory[a] = result


once = set()
def ADC(state, a) -> [(0x61, indx), (0x71, indy_read), (0x65, zpg), (0x69, imm), (0x6D, abs_read)]:
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

def SBC(state, a) -> [(0xE1, indx), (0xE5, zpg), (0xE9, imm), (0xED, abs_read), (0xF1, indy_read)]:
    a ^= 0xFF
    result = state.A + a + state.status_register['Carry']

    Z_set(state, byte(result))
    N_set(state, byte(result))
    state.status_register['Carry'] = 1*(result & 0xFF00 > 0)

    v1 =  ~(state.A ^ a) & (state.A ^ result) & 0x0080
    state.status_register['Overflow'] = 1*(v1 > 0)

    state.A = byte(result)

def STA(state, a) -> [(0x81, indx_write), (0x8D, imm2), (0x85, imm), (0x91, indy), (0x99, absy)]:
    state.memory[a] = state.A
def STX(state, a) -> [(0x86, imm), (0x8E, imm2)]: state.memory[a] = state.X
def STY(state, a) -> [(0x84, imm), (0x8C, imm2)]: state.memory[a] = state.Y
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

def BRK(state, a):
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
byte_counts = {_: 1, imm: 2, imm2: 3, zpg: 2, zpgx: 2, abs_read: 3, abs_x_read: 3,
               indx: 2, indx_write: 2, indy: 2, indy_read: 2, ind: 3, absy: 3}
for name, func in locals().copy().items():
    if settings := hasattr(func, '__annotations__') and func.__annotations__.get('return'):
        for opcode, addressing in settings:
            instructions[opcode] = (func, byte_counts[addressing], addressing)
