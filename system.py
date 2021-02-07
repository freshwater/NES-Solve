
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

        self.N = 0
        self.O = 0
        self.U = 1
        self.B = 0
        self.D = 0
        self.I = 1
        self.Z = 0
        self.C = 0

        # self.status_register = {
        #     'Negative': 0,
        #     'Overflow': 0,
        #     'Unused': 1,
        #     'Break': 0,
        #     'Decimal': 0,
        #     'Interrupt': 1,
        #     'Zero': 0,
        #     'Carry': 0,
        # }

        self.reset()

    def status_register_byte(self):
        # sr = self.status_register
        # status_register = (
        #     (sr['Negative'] << 7) + (sr['Overflow'] << 6) + (sr['Unused'] << 5) + (sr['Break'] << 4) +
        #     (sr['Decimal'] << 3) + (sr['Interrupt'] << 2) + (sr['Zero'] << 1) + (sr['Carry'] << 0))

        status_register = (self.N<<7) + (self.O<<6) + (self.U<<5) + (self.B<<4) + (self.D<<3) + (self.I<<2) + (self.Z<<1) + (self.C<<0)

        return status_register

    def status_register_byte_set(self, byte):
        # self.status_register = {
        #     key: (sr >> bit) & 0x01
        #     for key, bit in zip(self.status_register.keys(), [7, 6, 5, 4, 3, 2, 1, 0])}
        self.N = (byte>>7) & 0x01
        self.O = (byte>>6) & 0x01
        self.U = (byte>>5) & 0x01
        self.B = (byte>>4) & 0x01
        self.D = (byte>>3) & 0x01
        self.I = (byte>>2) & 0x01
        self.Z = (byte>>1) & 0x01
        self.C = (byte>>0) & 0x01

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
        condition2 = instruction[0].__name__ == 'SBC' and self.last_executed_opcode == 0xEB
        condition3 = instruction[0].__name__ == 'NOP' and self.last_executed_opcode != 0xEA
        prefix = '*' if condition1 or condition2 or condition3 else ''
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
        addressing = instructions.instructions[opcode][2]

        if addressing == instructions.implied:
            return ""
        elif addressing == instructions.immediate:
            return f'#${data1:02X}'

        elif addressing == instructions.absolute_address:
            return f'${data2*0x100 + data1:04X}'
        elif addressing == instructions.absolute_address_dereference:
            return f'(${data2*0x100 + data1:04X})'
        elif addressing == instructions.relative_address:
            return f'${program_counter + 2 + np.int8(data1):04X}'

        elif addressing in [instructions.absolute_dereference, instructions.absolute_address]:
            return f'${data2*0x0100 + data1:04X}'
        elif addressing in [instructions.zeropage_dereference, instructions.zeropage_address]:
            return f'${data1:02X}'

        elif addressing in [instructions.absolute_x_dereference, instructions.absolute_x_address]:
            return f'${data2*0x0100 + data1:04X},X'
        elif addressing in [instructions.zeropage_x_dereference, instructions.zeropage_x_address]:
            return f'${data1:02X},X'
        elif addressing in [instructions.indirect_x_dereference, instructions.indirect_x_address]:
            return f'(${data1:02X},X)'

        elif addressing in [instructions.absolute_y_dereference, instructions.absolute_y_address]:
            return f'${data2*0x0100 + data1:04X},Y'
        elif addressing in [instructions.zeropage_y_dereference, instructions.zeropage_y_address]:
            return f'${data1:02X},Y'
        elif addressing in [instructions.indirect_y_dereference, instructions.indirect_y_address]:
            return f'(${data1:02X}),Y'

        else:
            assert None, None


    def log_format(opcode, line):
        _, byte_count, addressing = instructions.instructions[opcode]

        if addressing == instructions.implied:
            return line[:2+byte_count] + line[-5:]
        else:
            return line[:3+byte_count] + line[-5:]


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

                import instructions as ins
                import region

                # print(operation.__name__, end=' * ')

                if addressing in [ins.indirect_x_address, ins.absolute_address, ins.zeropage_address,
                                  ins.indirect_y_address, ins.zeropage_x_address, ins.absolute_y_address,
                                  ins.absolute_x_address]:
                    print('_')
                    value1 = 0
                    address1 = addressing(self, data1, data2)

                else:
                    value1 = addressing(self, data1, data2)
                    address1 = region.ComputationState.NULL_ADDRESS

                if operation.__name__ in ['JSR', 'RTS', 'RTI']:
                    operation(self, address1)

                elif operation.__name__ in ['ADC', 'SBC']:
                    operation(self, value1)

                else:
                    # operation(self, value1)
                    # print(operation.__name__, addressing.__name__, value1, f'{address1:04X}')
                    region1 = operation(1, 1)

                    if byte_count == 1:
                        region1.transition(self, region.ComputationState())
                    else:
                        region1.transition(self, region.ComputationState(value1=value1, address=address1))

                self.operations_count += 1

                # diagnostic
                self.last_executed_opcode = opcode
                self.last_executed_data1 = data1
                self.last_executed_data2 = data2


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