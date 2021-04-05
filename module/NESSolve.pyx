
# distutils: language = c++

cimport NESSolveModule
import numpy as np
cimport numpy as npc

def run(str file_location,
        npc.ndarray[char, ndim=2] actions,
        int num_blocks):

    if len(file_location) == 0:
        return

    frame_size_per_channel = 256*240
    num_instances = actions.shape[0]
    num_actions = actions.shape[1]
    actions = actions.transpose().copy(order='C')

    cdef const unsigned char[:] c_file_location = file_location.encode()

    array = np.empty(num_instances*frame_size_per_channel, dtype=np.uint8)
    cdef char[::1] frames = array

    cdef char[:,::1] c_actions = actions

    NESSolveModule.run(&c_file_location[0], len(file_location),
                       &c_actions[0][0], num_instances, num_actions,
                       num_blocks,
                       &frames[0])

    return array
