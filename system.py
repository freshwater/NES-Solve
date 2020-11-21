
import numpy as np

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
        self.program_counter = self.memory[RESET_VECTOR_H]*0x100 + self.memory[RESET_VECTOR_L]
        # self.program_counter = 0xC000
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
        self.array[0x8000:0x8000+len(program_data)] = program_data
        # self.array[0xC000:0xC000+len(program_data)] = program_data

    def __getitem__(self, address):
        result = self.array[address & 0xFFFF]

        Behaviors.reset_vblank_on_read(self.state, address)

        return result

    def __setitem__(self, address, value):
        self.array[address] = value

        # if 0x2000 <= address & address <= 0x2008:
        #     print(f'write {address:04X} {value:08b}')