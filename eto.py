
import json
import numpy as np

from operations import *

with open('data/g.json', 'rb') as file:
    data = json.loads(file.read())

program_data = np.array(data['Program'], dtype=np.uint8)
character_data = np.array(data['Character'], dtype=np.uint8)


##  begin

state = {
    'ProgramCounter': 0,
    'Accumulator': 0, 'X': 0, 'Y': 0,
    'StackRegister': 0,
    'StatusRegister': {
        'Negative': 0,
        'Zero': 0,
        'Decimal': 0,
        'Interrupt': 0,
        'Carry': 0,
        'Overflow': 0
    }
}

memory = np.zeros(0x10000, dtype=np.uint8)
memory[0x8000:0x8000+len(program_data)] = program_data

state['Memory'] = memory

##

def reset():
    state['ProgramCounter'] = state['Memory'][0xFFFD]*0x0100 + state['Memory'][0xFFFC]
    state['StackZero'] = 0x0100
    state['StackOffset'] = 0xFD

def increment():
    opcode = state['Memory'][state['ProgramCounter']]
    state['ProgramCounter'] += 1

    return opcode

def vertical_blank(): state['Memory'][0x2002] = 0x80

ops = {}

def visit(index, text):
    count, text = ops.get(index, (0, text))
    ops[index] = (count + 1, text)

vertical_blank()
reset()

indexes = []

def do(n):
    global last

    for _1 in range(n):
        index = state['ProgramCounter']
        opcode = increment()
        indexes.append(index)

        operation, byte_count, addressing = instructions[opcode.item()]

        if byte_count == 1:
            visit(index, f'{opcode:02X} {operation.__name__}')
            operation(state, 0)

        elif byte_count == 2:
            data = increment()

            fy = ",y" if addressing == indy else ""
        
            visit(index, f'{opcode:02X} {operation.__name__} .${data:02x}' + fy)
            operation(state, addressing(state, data))

        elif byte_count == 3:
            data1, data2 = increment(), increment()

            fy = ",y" if addressing == absy else ""
            visit(index, f'{opcode:02X} {operation.__name__} ${data2:02x}{data1:02x}{fy} ;{data1*0x0100 + data2:04x}')
            operation(state, addressing(state, data1, data2))

        else:
            1 / 0

do(12708 + 6611 + 1000)
opcode = state['Memory'][state['ProgramCounter']]

b = {'False': " ", '0': '●', '1': '•', '2': '⋅', '3': '⋅'}
a = lambda i: b[str((i in indexes[-4:]) and indexes[-4:][::-1].index(i))]
ops = sorted(ops.items())
print('\n' + '\n'.join([f'{a(index)}{count:2} {index:02X} {op}' for index, (count, op) in ops]) + '\n')

print()
print(hex(state['ProgramCounter']), state, hex(opcode.item()), opcode.item())
print()

# print(f":{state['Memory'][0x04A0]:X}")
# print(hex(248))