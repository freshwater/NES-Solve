
import json
import numpy as np

from operations import *

with open('data/g.json', 'rb') as file:
    data = json.loads(file.read())

program_data = np.array(data['Program'], dtype=np.uint8)
character_data = np.array(data['Character'], dtype=np.uint8)


##  begin

ops = {}

def visit(index, text):
    count, text = ops.get(index, (0, text))
    ops[index] = (count + 1, text)

def increment():
    opcode = state.memory[state.program_counter]
    state.program_counter += 1

    return opcode

indexes = []

# 341 PPU cycles per scanline
# 113.667 CPU cycles per scanline
# 261 scanlines
# 241-261 is vblank

PPU_CONTROL = 0x2000
PPU_STATUS = 0x2002

def do(n):
    for _1 in range(999999999):
        for v in range(-1, 261):
            vblank = None

            for h in range(341):

                if vblank == None and v == 241:
                    if h == 2:
                        state.memory[PPU_STATUS] = 0x80

                        if state.memory[PPU_CONTROL] & 0x80:
                            # return n, v, h
                            NMI(state)

                        vblank = True

                n -= 1
                if n == 0:
                    return _1, v, h

                if n % 3 == 0:
                    index = state.program_counter
                    opcode = increment()
                    indexes.append(index)

                    operation, byte_count, addressing = instructions[opcode.item()]
                    visit_data = {'opcode': opcode, 'byte_count': byte_count,
                                  'operation': operation.__name__, 'addressing': addressing}

                    if byte_count == 1:
                        # here. merge dicts
                        visit(index, visit_data)
                        operation(state, 0)

                    elif byte_count == 2:
                        data = increment()

                        fy = ",y" if addressing == indy else ""

                        visit(index, visit_data | {'data': data})
                        operation(state, addressing(state, data))

                    elif byte_count == 3:
                        data1, data2 = increment(), increment()

                        fy = ",y" if addressing == absy else ""
                        visit(index, visit_data | {'data': data2*0x0100 + data1})
                        operation(state, addressing(state, data1, data2))

                    else:
                        1 / 0

state = State(program_data)

labels = {
    "0778": "MIRROR_PPU_CTRL_1",
    "2000": "PPU_CTRL_1",
    "2001": "PPU_CTRL_2",
    "2002": "PPU_STATUS",
    "2006": "PPU_ADDRESS",
    "2007": "PPU_DATA",
    "4014": "SPRITE",
    "8082": "NMI",
    "809E": "ScreenOff",
    "8E92": "WriteBuffer",
    "8EDD": "UpdateScreen",
    "8EED": "WritePPUReg1",
    "90CC": "InitializeMemory",
}

labels = {int(key, 16): value for key, value in labels.items()}
l = lambda i: f'{labels.get(i, ""):15}'

import sys
_1, v, h = do(int(sys.argv[1]))

def r(byte_count, data, addressing):
    if data != None:
        if byte_count == 1:
            return ''
        elif byte_count == 2:
            if addressing == indy:
                return f'${data:02X},y'
            else:
                return f'${data:02X}'
        elif byte_count == 3:
            return  f'${data:04X} {labels.get(data, "")}'
    else:
        return ''

b = {'False': " ", '0': '●', '1': '•', '2': '⋅', '3': '⋅'}
a = lambda i: b[str((i in indexes[-4:]) and indexes[-4:][::-1].index(i))]
ops = sorted(ops.items())
print('\n' + '\n'.join([
    f'{l(index)}{a(index)}{count:4} {index:02X} .{d["opcode"]:02X} {d["operation"]} {r(d["byte_count"], d.get("data"), d["addressing"])}'
    for index, (count, d) in ops]) + '\n')

print()
# opcode = state['Memory'][state['ProgramCounter']]
opcode = state.memory[state.program_counter]
print(hex(state.program_counter), state, hex(opcode.item()), opcode.item())
print()
print("i", _1, "v", v, "h", h)
print()

# print(f":{state['Memory'][0x04A0]:X}")
# print(hex(248))