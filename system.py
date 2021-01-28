
import numpy as np
import instructions

#-

PPU_CONTROL_1 = 0x2000
PPU_REGISTERS = 0x2000
PPU_CONTROL_2 = 0x2001
PPU_STATUS = 0x2002

PPU_OAM_ADDRESS = 0x2003
PPU_OAM_DATA = 0x2004
PPU_OAM_DMA = 0x4014

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

class State:
    def __init__(self, rom_data, program_counter=None, load_point=None):
        self.program_counter = None
        self.program_counter_initial = program_counter
        self.operation_countdown = None
        self.A = 0
        self.X = 0
        self.Y = 0
        self.stack_offset = STACK_OFFSET_INITIAL

        self.cycle = 0
        self.operation_countdown = 7
        self.vertical_scan = -1
        self.horizontal_scan = 0
        self.vertical_blank = None

        self.operations_count = 0
        self.last_executed_opcode = None

        program_data = np.array(rom_data['Program'], dtype=np.uint8)

        self.ppu = PictureProcessingUnit()
        self.memory = Memory(self, program_data, load_point=load_point)

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

    def reset(self):
        if self.program_counter_initial:
            self.program_counter = self.program_counter_initial
        else:
            self.program_counter = self.memory[RESET_VECTOR_H]*0x100 + self.memory[RESET_VECTOR_L]

        self.stack_offset = STACK_OFFSET_INITIAL

    def timings_string(self):
        return f'[cycle[{self.cycle}], operations[{self.operations_count}], h[{self.horizontal_scan}], v[{self.vertical_scan}]]'

    def state_string(self):
        byte_count = 1 + bool(self.last_executed_data1) + bool(self.last_executed_data2)

        # code = f"""{self.program_counter - byte_count:04X} {self.last_executed_opcode:02X} """

        code = f"""{self.previous_program_counter:04X} {self.last_executed_opcode:02X} """

        if self.last_executed_data1 != None:
            code += f' {self.last_executed_data1:02X}'
        if self.last_executed_data2 != None:
            code += f' {self.last_executed_data2:02X}'

        instruction = instructions.instructions[self.last_executed_opcode]
        condition1 = instruction[0].__name__ in ['LAX', 'SAX', 'DCP', 'ISB', 'SLO', 'RLA', 'SRE', 'RRA']
        condition2 = self.last_executed_opcode in [0x04, 0x0C, 0x44, 0x64, 0xEB]
        prefix = '*' if condition1 or condition2 else ''
        code += f' {prefix}{instruction[0].__name__[:3]}'

        data = State.instruction_format(self.last_executed_opcode,
                                        self.last_executed_data1,
                                        self.last_executed_data2,
                                        self.previous_program_counter)

        code += ' ' + data

        A, X, Y = self.previous_AXY
        code += f' A:{A:02X} X:{X:02X} Y:{Y:02X}'
        code += f' P:{self.previous_status:02X}'
        code += f' SP:{self.previous_stack_offset:02X}'

        code = ' '.join(code.split())

        return code


    def instruction_format(opcode, data1, data2, program_counter):
        byte_count = 1 + (data1 != None) + (data2 != None)
        addressing = instructions.instructions[opcode][2]

        if opcode in [0x5B, 0x5F, 0x63, 0xA3, 0xA7, 0xAF, 0xB3]:
            if addressing == instructions.absx_write:
                return f'${data2*0x0100 + data1:04X},X'
            elif addressing in [instructions.indx, instructions.indx_write]:
                return f'(${data1:02X},X)'
            elif addressing == instructions.indy_read:
                return f'(${data1:02X}),Y'
            elif addressing == instructions.zpg:
                return f'${data1:02X}'
            elif addressing == instructions.absy_write:
                return f'${data2*0x0100 + data1:04X},Y'
            elif addressing == instructions.abs_read:
                return f'${data2*0x0100 + data1:04X}'
            else:
                return "NOPEOUEU"
        elif opcode in [0x15, 0x16, 0x17, 0x35, 0x36, 0x37, 0x55, 0x57, 0x56, 0x75, 0x76,
                        0x94, 0x95, 0xB4, 0xB5, 0xD5, 0xD6, 0xD7, 0xF5, 0xF6, 0xF7]:
            return f'${data1:02X},X'
        elif opcode in [0x1D, 0x1E, 0x1F, 0x3D, 0x3E, 0x3F, 0x5D, 0x5E, 0x7D, 0x7E, 0x9D, 0xBC,
                        0xBD, 0xDD, 0xDE, 0xDF, 0xFD, 0xFE, 0xFF]:
            return f'${data2*0x0100 + data1:04X},X'
        elif opcode in [0x01, 0x03, 0x21, 0x23, 0x41, 0x43, 0x61, 0x81, 0xA1, 0xC1, 0xE1,
                        0x83, 0xC3, 0xE3]:
            return f'(${data1:02X},X)'
        elif opcode in [0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0]:
            return f'${program_counter + 2 + np.int8(data1):04X}'
        elif opcode in [0x11, 0x13, 0x31, 0x33, 0x51, 0x53, 0x71, 0x91, 0xB1, 0xD1, 0xF1,
                        0xD3, 0xF3]:
            return f'(${data1:02X}),Y'
        elif opcode in [0x19, 0x1B, 0x39, 0x3B, 0x59, 0x79, 0x99, 0xB9, 0xBE, 0xD9, 0xF9,
                        0xBF, 0xDB, 0xFB]:
            return f'${data2*0x0100 + data1:04X},Y'
        elif opcode == 0x6C:
            return f'(${data2*0x0100 + data1:04X})'
        elif opcode in [0x96, 0xB6, 0xB7, 0x97]:
            return f'${data1:02X},Y'
        elif opcode in [0x09, 0x29, 0x49, 0x69, 0xC9, 0xA0, 0xA2, 0xA9, 0xC0,
                        0xE0, 0xEB, 0xE9]:
            return f'#${data1:02X}'
        elif opcode in [0x80]:
            return f'#${data1:02X}'

        if byte_count == 1:
            return ""
        elif byte_count == 2:
            return f'${data1:02X}'
        else:
            return f'${data2*0x0100 + data1:04X}'

    def log_format(opcode, line):
        byte_count = instructions.instructions[opcode][1]

        if opcode in [0x04, 0x0C, 0x3F, 0x44, 0x53, 0x57, 0x5B, 0x5F, 0x64, 0xA3,
                      0xA7, 0xAF, 0xB3]:
            return line[:3+byte_count] + line[-5:]

        elif opcode in [0x0D, 0x0E, 0x2C, 0x2D, 0x2E, 0x4D, 0x4E,
                      0x6D, 0x6E, 0x8C, 0x8D, 0x8E,
                      0xAC, 0xAD, 0xAE,
                      0xCC, 0xCD, 0xCE, 0xEC, 0xED, 0xEE]:
            return line[:6] + line[8:]

        elif opcode in [0x05, 0x06, 0x25, 0x26, 0x45, 0x46,
                        0x65, 0x66, 0x84, 0x86, 0xA4,
                        0xA5, 0xA6, 0xC4, 0xC5, 0xC6, 0xE4, 0xE5, 0xE6]:
            return line[:5] + line[7:]
        elif opcode in [0x15, 0x16, 0x35, 0x36, 0x37, 0x55, 0x56, 0x75, 0x76,
                        0x94, 0x95, 0xB4, 0xB5, 0xD5, 0xD6, 0xF5, 0xF6]:
            return line[:5] + line[9:]
        elif opcode in [0x1D, 0x1E, 0x3D, 0x3E, 0x5D, 0x5E, 0x7D, 0x7E, 0x9D, 0xBC,
                        0xBD, 0xDD, 0xDE, 0xFD, 0xFE]:
            return line[:6] + line[10:]
        elif opcode in [0x01, 0x21, 0x41, 0x61, 0x81, 0xC1, 0xE1]:
            return line[:5] + line[11:]
        elif opcode in [0x0A, 0x2A, 0x4A, 0x6A]:
            return line[:3] + line[4:]
        elif opcode in [0x11, 0x31, 0x51, 0x71, 0x91, 0xB1, 0xD1, 0xF1]:
            return line[:5] + line[11:]
        elif opcode in [0x19, 0x39, 0x3B, 0x59, 0x79, 0x99, 0xB9, 0xBE, 0xD9, 0xF9]:
            return line[:6] + line[10:]
        elif opcode == 0x6C:
            return line[:6] + line[8:]
        elif opcode in [0x96, 0xB6]:
            return line[:5] + line[9:]

        elif opcode in [0x1A, 0x3A, 0x5A, 0x7A, 0xDA, 0xFA]:
            i = line.index("*NOP")
            return line[:i] + ["NOP"] + line[i+1:]

        elif opcode in [0x80]:
            i = line.index("*NOP")
            return line[:i] + ["NOP"] + line[i+1:]

        elif opcode in [0x14, 0x34, 0x54, 0x74, 0xD4, 0xF4,
                        0x1C, 0x3C, 0x5C, 0x7C, 0xDC, 0xFC]:
            i = line.index("*NOP")
            return line[:i] + ["NOP", line[i+1][:line[i+1].index(',')]] + line[i+6:]

        elif opcode in [0xB7, 0xBF]:
            i = line.index("*LAX")
            return line[:i+2] + line[i+6:]

        elif opcode in [0x97]:
            i = line.index("*SAX")
            return line[:i+2] + line[i+6:]
        
        elif opcode in [0xD3]:
            i = line.index("*DCP")
            return line[:i+2] + line[i+8:]
        elif opcode in [0xD7, 0xDB, 0xDF]:
            i = line.index("*DCP")
            return line[:i+2] + line[i+6:]
        elif opcode in [0xF7, 0xFB, 0xFF]:
            i = line.index("*ISB")
            return line[:i+2] + line[i+6:]
        elif opcode in [0xF3]:
            i = line.index("*ISB")
            return line[:i+2] + line[i+8:]
        elif opcode in [0x13]:
            i = line.index("*SLO")
            return line[:i+2] + line[i+8:]
        elif opcode in [0x17, 0x1B, 0x1F]:
            i = line.index("*SLO")
            return line[:i+2] + line[i+6:]
        elif opcode in [0x33]:
            i = line.index("*RLA")
            return line[:i+2] + line[i+8:]

        elif "@" in line:
            return line[:line.index("@")] + line[line.index("@")+6:]

        elif "*NOP" in line:
            i = line.index("*NOP")
            return line[:i] + ["NOP", line[i+1]] + line[i+5:]
        elif "=" in line:
            return line[:line.index("=")] + line[line.index("=")+2:]

        return line



    def next(self):
        self.cycle += 1
        self.horizontal_scan += 1

        if self.horizontal_scan == 341:
            self.horizontal_scan = 0
            self.vertical_scan += 1

        if self.vertical_blank and self.vertical_scan == 241:
            if self.horizontal_scan == 2:
                self.memory[PPU_STATUS] = 0x80

                if state.ppu.registers.array[PPU_CONTROL_1 - PPU_CONTROL_1] & 0x80:
                    if state.ppu.registers.array[PPU_STATUS - PPU_CONTROL_1] & 0x80:
                        NMI(state)

                self.vertical_blank = True

        if self.vertical_scan == 261:
            self.vertical_scan = -1

        # operation
        if self.cycle % 3 == 0:
            opcode = self.memory[self.program_counter]
            self.operation_countdown -= 1

            if self.operation_countdown == 0:
                operation, byte_count, addressing = instructions.instructions[opcode]
                data1 = self.memory[self.program_counter+1] if byte_count > 1 else None
                data2 = self.memory[self.program_counter+2] if byte_count > 2 else None

                self.previous_program_counter = self.program_counter
                self.program_counter += byte_count
                self.operation_countdown = 3

                # diagnostic
                self.previous_AXY = [self.A, self.X, self.Y]
                self.previous_status = self.status_register_byte()
                self.previous_stack_offset = self.stack_offset

                operation(self, addressing(self, data1, data2))

                self.operations_count += 1

                # diagnostic
                self.last_executed_opcode = opcode
                self.last_executed_data1 = data1
                self.last_executed_data2 = data2

                # print(f'[op[{operation.__name__}], bytes[{byte_count}], addr[{addressing.__name__}]]')


class PictureProcessingUnit:
    def __init__(self):
        """
        https://wiki.nesdev.com/w/index.php/PPU_registers
        """

        self.registers = PictureProcessingUnit.Registers(ppu=self)
        self.address_latch = 0
        self.address = None
        self.dma = np.array([0], dtype=np.uint8)
        self.array = np.zeros(0x4000, dtype=np.uint8)

    def dma_read(self):
        return self.dma[0]

    def dma_write(self, value, memory_array):
        """
        http://wiki.nesdev.com/w/index.php/PPU_registers#OAMDMA
        This port is located on the CPU. Writing $XX will upload 256 bytes of data
        from CPU page $XX00-$XXFF to the internal PPU OAM. This page is typically
        located in internal RAM, commonly $0200-$02FF, but cartridge RAM or ROM
        can be used as well.
        """

        dma_address = self.registers.array[PPU_OAM_ADDRESS - PPU_REGISTERS]
        self.dma[0] = value
        data = memory_array[value*0x0100:value*0x0100+0x0100]
        self.array[dma_address:dma_address+0x0100] = data

        print('__ ' + str(data.reshape(-1, 0x10).shape))
        for row in data.reshape(-1, 8):
            print('__ ' + ' '.join([f'{j:02X}' if j != 0 else '--' for j in row]))

    class Registers:
        def __init__(self, ppu):
            self.ppu = ppu

            self.array = np.zeros(8, dtype=np.uint8)

        def control_register_1(self):
            """
            7  bit  0
            ---- ----
            VPHB SINN
            |||| ||||
            |||| ||++- Base nametable address
            |||| ||    (0 = $2000; 1 = $2400; 2 = $2800; 3 = $2C00)
            |||| |+--- VRAM address increment per CPU read/write of PPUDATA
            |||| |     (0: add 1, going across; 1: add 32, going down)
            |||| +---- Sprite pattern table address for 8x8 sprites
            ||||       (0: $0000; 1: $1000; ignored in 8x16 mode)
            |||+------ Background pattern table address (0: $0000; 1: $1000)
            ||+------- Sprite size (0: 8x8 pixels; 1: 8x16 pixels)
            |+-------- PPU master/slave select
            |          (0: read backdrop from EXT pins; 1: output color on EXT pins)
            +--------- Generate an NMI at the start of the
                    vertical blanking interval (0: off; 1: on)
            """
            return {
                'NMIOnVBlank': self.registers[0] >> 7 & 0x01,
                # 'PPU_Unused': self.registers[0] >> 6 & 0x01,
                'SpriteSize': self.registers[0] >> 5 & 0x01,
                'BackgroundPatternTable': self.registers[0] >> 4 & 0x01,
                'SpritePatternTable': self.registers[0] >> 3 & 0x01,
                'VRAMDataIncrement': self.registers[0] >> 2 & 0x01,
                'Nametable': self.registers[0] >> 0 & 0x03
            }

        def __setitem__(self, register_index, value):
            self.array[register_index] = value

            if PPU_ADDRESS - PPU_REGISTERS == register_index:
                if self.ppu.address_latch == 0:
                    self.ppu.address = value * 0x0100
                elif self.ppu.address_latch == 1:
                    self.ppu.address += value
                    # print(f'WRITE-{self.ppu.address:04X}')
                else:
                    assert False

                self.ppu.address_latch ^= 1

            elif PPU_OAM_ADDRESS - PPU_REGISTERS == register_index:
                print("__-DECODE", hex(value))



        def __getitem__(self, register_index):
            result = self.array[register_index]

            Behaviors.reset_vblank_on_read(self.ppu, register_index)

            return result


class Memory:
    def __init__(self, state, program_data, load_point=None):
        load_point = load_point or 0x8000
        self.state = state

        self.array = np.zeros(0x10000, dtype=np.uint8)
        self.array[load_point:load_point+len(program_data)] = program_data

    def __getitem__(self, address):
        """
        https://wiki.nesdev.com/w/index.php/PPU_registers
        The PPU exposes eight memory-mapped registers to the CPU.
        These nominally sit at $2000 through $2007 in the CPU's address space,
        but because they're incompletely decoded, they're mirrored in every 8 bytes
        from $2008 through $3FFF, so a write to $3456 is the same as a write to $2006.
        """
        if 0x2000 <= address and address <= 0x3FFF:
            result = self.state.ppu.registers[address % 8]
        elif 0x4014 == address:
            result = self.state.ppu.dma_read()
        else:
            result = self.array[address & 0xFFFF]

        return result

    def __setitem__(self, address, value):
        if 0x2000 <= address and address <= 0x3FFF:
            self.state.ppu.registers[address % 8] = value
        elif 0x4014 == address:
            """
            http://wiki.nesdev.com/w/index.php/PPU_registers#OAMDMA
            The CPU is suspended during the transfer, which will take 513 or 514 cycles after
            the $4014 write tick. (1 wait state cycle while waiting for writes to complete,
            +1 if on an odd CPU cycle, then 256 alternating read/write cycles.)
            """
            result = self.state.ppu.dma_write(value, memory_array=self.array)
            self.state.operation_countdown = 513
        else:
            self.array[address] = value

        # if 0x2000 <= address & address <= 0x2008:
        #     print(f'write {address:04X} {value:08b}')