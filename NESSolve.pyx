
# distutils: language = c++

cimport NESSolveModule
import numpy as np

def run(str file_location):
    if len(file_location) == 0:
        return

    frame_size_per_channel = 256*240

    cdef const unsigned char[:] c_file_location = file_location.encode()
    array = np.empty(frame_size_per_channel, dtype=np.uint8)
    cdef char[::1] frames = array

    NESSolveModule.run(&c_file_location[0], len(file_location), &frames[0])

    return array
