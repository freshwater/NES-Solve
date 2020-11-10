
import json
import numpy as np

from operations import *


##  begin

ops = {}
log = []

def visit(index, text):
    log.append((index, text))
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

                if True or n % 3 == 0:
                    index = state.program_counter
                    opcode = increment()
                    indexes.append(index)

                    sr = state.status_register

                    operation, byte_count, addressing = instructions[opcode.item()]
                    visit_data = {'opcode': opcode, 'byte_count': byte_count,
                                  'operation': operation.__name__, 'addressing': addressing,
                                  'A': state.A, 'X': state.X, 'Y': state.Y,
                                  'status_register': state.status_register_byte(),
                                  'stack_pointer': state.stack_offset}

                    if byte_count == 1:
                        # here. merge dicts
                        visit(index, visit_data)
                        operation(state, 0)

                    elif byte_count == 2:
                        data = increment()

                        fy = ",y" if addressing == indy else ""

                        visit(index, visit_data | {'data': data, 'data1': data})
                        operation(state, addressing(state, data))

                    elif byte_count == 3:
                        data1, data2 = increment(), increment()

                        fy = ",y" if addressing == absy else ""
                        visit(index, visit_data | {'data': data2*0x0100 + data1, 'data1': data1, 'data2': data2})
                        operation(state, addressing(state, data1, data2))

                    else:
                        1 / 0


#-


# with open('data/g.json', 'rb') as file:
# with open('data/full_palette.json', 'rb') as file:
with open('data/nestest.json', 'rb') as file:
    data = json.loads(file.read())

with open('data/nestest_AXYPSP.log') as file:
    log_k = file.readlines()

program_data = np.array(data['Program'], dtype=np.uint8)
character_data = np.array(data['Character'], dtype=np.uint8)

state = State(program_data)

import sys
_1, v, h = do(int(sys.argv[1]) + 1)

def to_line(op):
    return op

# for l1, l2 in zip()
print()
print(_1, v, h)
print()

for i, ((index, op), line_k) in enumerate(zip(log, log_k), 1):
    line_k = line_k.split()

    line = [f'{index:04X}', f'{op["opcode"]:02X}']

    if op['byte_count'] > 1:
        line.append(f'{op["data1"]:02X}')

    if op['byte_count'] > 2:
        line.append(f'{op["data2"]:02X}')

    line.append(f'{op["operation"][:3]}')

    if False:
        pass

    elif op["opcode"] == 0x10: #BPL
        line.append(f'${index + 2 + np.int8(op["data1"]):04X}')
    elif op["opcode"] == 0x24: #BIT
        line.append(f'${op["data"]:02X}')
        line_k = line_k[:5] + line_k[7:]
    elif op["opcode"] in [0x0A, 0x2A, 0x4A, 0x6A]: # ASL, ROL, LSR, ROR
        line_k = line_k[:3] + line_k[4:]
    elif op["opcode"] == 0x8E: #STX
        line.append(f'${op["data"]:04X}')
        line_k = line_k[:6] + line_k[8:]

    elif op["opcode"] in [0x05, 0x06, 0x25, 0x26, 0x45, 0x46, 0x65, 0x66, 0x84, 0x86, 0xA4,
                          0xA5, 0xA6, 0xC4, 0xC5, 0xC6, 0xE4, 0xE5, 0xE6]:
        line.append(f'${op["data"]:02X}')
        line_k = line_k[:5] + line_k[7:]

    elif op["opcode"] in [0x15, 0x16, 0x35, 0x36, 0x55, 0x56, 0x75, 0x76,
                          0x94, 0x95, 0xB4, 0xB5, 0xD5, 0xD6, 0xF5, 0xF6]:
        line.append(f'${op["data"]:02X},X')
        line_k = line_k[:5] + line_k[9:]

    elif op["opcode"] in [0x96, 0xB6]:
        line.append(f'${op["data"]:02X},Y')
        line_k = line_k[:5] + line_k[9:]

    elif op["opcode"] in [0x01, 0x21, 0x41, 0x61, 0x81, 0xC1, 0xE1]:
        line.append(f'(${op["data"]:02X},X)')
        line_k = line_k[:5] + line_k[11:]

    elif op["opcode"] in [0x1D, 0x3D, 0x5D, 0x7D, 0x9D, 0xBC, 0xBD, 0xDD, 0xFD]:
        line.append(f'${op["data"]:04X},X')
        line_k = line_k[:6] + line_k[10:]

    elif op["opcode"] in [0x19, 0x39, 0x59, 0x79, 0x99, 0xB9, 0xD9, 0xF9]:
        line.append(f'${op["data"]:04X},Y')
        line_k = line_k[:6] + line_k[10:]

    elif op["opcode"] in [0x11, 0x31, 0x51, 0x71, 0x91, 0xB1, 0xD1, 0xF1]:
        line.append(f'(${op["data"]:02X}),Y')
        line_k = line_k[:5] + line_k[11:]


    elif op["opcode"] in [0x0D, 0x0E, 0x2C, 0x2D, 0x2E, 0x4D, 0x4E, 0x6D, 0x6E, 0x8C, 0x8D,
                          0xAC, 0xAD, 0xCC, 0xCD, 0xCE, 0xEC, 0xED, 0xEE]:
        line.append(f'${op["data"]:04X}')
        line_k = line_k[:6] + line_k[8:]
    elif op["opcode"] in [0x6C]:
        line.append(f'(${op["data"]:04X})')
        line_k = line_k[:6] + line_k[8:]

    elif op["opcode"] == 0xA1: #LDA
        line_k = line_k[:4] + line_k[11:]
    elif op["opcode"] == 0xAE: #LDX
        line.append(f'${op["data"]:04X}')
        line_k = line_k[:6] + line_k[8:]
    elif op["opcode"] == 0x30: # BMI
        line.append(f'${index + 2 + np.int8(op["data1"]):04X}')
    elif op["opcode"] == 0x50: # BVC
        line.append(f'${index + 2 + np.int8(op["data1"]):04X}')
    elif op["opcode"] == 0x70: # BVS
        line.append(f'${index + 2 + np.int8(op["data1"]):04X}')
    elif op["opcode"] == 0x85: #STA
        # line.append(f'${op["data"]:02X} = {op["A"]:02X}')
        line.append(f'${op["data"]:02X} = __')
        line_k[6] = "__"
    elif op["opcode"] == 0x86: #STX
        line.append(f'${op["data"]:02X} = {op["X"]:02X}')
    elif op["opcode"] == 0x90: #BCC
        line.append(f'${index + 2 + np.int8(op["data1"]):04X}')
    elif op["opcode"] == 0xB0: #BCS
        line.append(f'${index + 2 + np.int8(op["data1"]):04X}')
    elif op["opcode"] == 0xD0: #BCS
        line.append(f'${index + 2 + np.int8(op["data1"]):04X}')
    elif op["opcode"] == 0xF0: #BEQ
        line.append(f'${index + 2 + np.int8(op["data1"]):04X}')
    else:
        if op['byte_count'] == 2:
            line.append(f'#${op["data"]:02X}')
        elif op['byte_count'] == 3:
            line.append(f'${op["data"]:4X}')

    line.append(f'A:{op["A"]:02X}')
    line.append(f'X:{op["X"]:02X}')
    line.append(f'Y:{op["Y"]:02X}')

    line.append(f'P:{op["status_register"]:02X}')
    line.append(f'SP:{op["stack_pointer"]:02X}')

    l1 = ' '.join(line)
    l2 = ' '.join(line_k)

    if l1 == l2:
        print('   ', l1)
    else:
        print(i)
        print('j >', l1)
        print('k >', l2)
        print()

        exit()

## labels = {
##     "0778": "MIRROR_PPU_CTRL_1",
##     "2000": "PPU_CTRL_1",
##     "2001": "PPU_CTRL_2",
##     "2002": "PPU_STATUS",
##     "2006": "PPU_ADDRESS",
##     "2007": "PPU_DATA",
##     "4014": "SPRITE",
##     "8082": "NMI",
##     "809E": "ScreenOff",
##     "8E92": "WriteBuffer",
##     "8EDD": "UpdateScreen",
##     "8EED": "WritePPUReg1",
##     "90CC": "InitializeMemory",
## }
## 
## labels = {int(key, 16): value for key, value in labels.items()}
## l = lambda i: f'{labels.get(i, ""):15}'
## 
## import sys
## _1, v, h = do(int(sys.argv[1]))
## 
## def r(byte_count, data, addressing):
##     if data != None:
##         if byte_count == 1:
##             return ''
##         elif byte_count == 2:
##             if addressing == indy:
##                 return f'${data:02X},y'
##             else:
##                 return f'${data:02X}'
##         elif byte_count == 3:
##             return  f'${data:04X} {labels.get(data, "")}'
##     else:
##         return ''
## 
## b = {'False': " ", '0': '●', '1': '•', '2': '⋅', '3': '⋅'}
## a = lambda i: b[str((i in indexes[-4:]) and indexes[-4:][::-1].index(i))]
## ops = sorted(ops.items())
## print('\n' + '\n'.join([
##     f'{l(index)}{a(index)}{count:4} {index:02X} {d["opcode"]:02X} {d["operation"]} {r(d["byte_count"], d.get("data"), d["addressing"])}'
##     for index, (count, d) in ops]) + '\n')
## 
## print()
## # opcode = state.memory[state.program_counter]
## # print(hex(state.program_counter), state, hex(opcode.item()), opcode.item())
## print("i", _1, "v", v, "h", h)
## print()
## 
## # print(f":{state['Memory'][0x04A0]:X}")
## # print(hex(248))