import json

import system
import instructions

import sys

file_name = sys.argv[1]
log_name = sys.argv[2]
cycles_max = int(sys.argv[3])

with open(file_name) as file:
    data = json.loads(file.read())

with open(log_name) as file:
    log = file.readlines()
    log_index = 0

state = system.State(rom_data=data, program_counter=0xC000, load_point=0xC000)

print(state.timings_string())

last_operations_count = 0
counter = 0
for i in range(cycles_max):
    state.next()

    if state.operations_count != last_operations_count:
        state_string = state.state_string()
        last_operations_count = state.operations_count

        line = log[log_index]
        log_index += 1

        line = line.split()
        line = line[:line.index('PPU:')]
        line = system.State.log_format(state.last_executed_opcode, line)
        line = ' '.join(line)

        if line == state_string:
            print('│', line)
        else:
            print('·', line)
            print(' ', state_string)
            counter += 1

print(counter)

print(state.timings_string())

