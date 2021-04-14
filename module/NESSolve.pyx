
# distutils: language = c++

cimport NESSolveModule
import numpy as np
cimport numpy as npc

def run(str file_location,
        npc.ndarray[char, ndim=2] actions,
        int num_blocks):

    if len(file_location) == 0:
        return

    cdef const unsigned char[:] c_file_location = file_location.encode()

    frame_size_per_channel = 256*240

    num_instances = actions.shape[0]
    num_actions = actions.shape[1]
    actions = actions.transpose().copy(order='C')
    cdef char[:,::1] c_actions = actions

    data_lines_size = num_instances*num_actions*(256//16)*(240//16)*4;
    array_data_lines = np.empty(data_lines_size // 4, dtype=np.float32);
    cdef float[::1] data_lines = array_data_lines

    array_red = np.empty(num_instances*frame_size_per_channel, dtype=np.uint8)
    array_green = np.empty(num_instances*frame_size_per_channel, dtype=np.uint8)
    array_blue = np.empty(num_instances*frame_size_per_channel, dtype=np.uint8)
    cdef char[::1] frames_red = array_red
    cdef char[::1] frames_green = array_green
    cdef char[::1] frames_blue = array_blue

    NESSolveModule.run(&c_file_location[0], len(file_location),
                       &c_actions[0][0], num_instances, num_actions,
                       num_blocks,
                       &data_lines[0],
                       &frames_red[0], &frames_green[0], &frames_blue[0])

    return array_data_lines, array_red, array_green, array_blue
