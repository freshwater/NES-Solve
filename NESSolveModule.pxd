
# distutils: language = c++

cdef extern from "NESSolveModule.cu" namespace "NESSolveModule":
    cdef void run(const unsigned char* file_location, int file_location_size, char* frames_red_out) except +
