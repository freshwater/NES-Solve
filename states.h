
struct Trace {
    int16u_t program_counter;
    int8u_t opcode;
    int8u_t byte1;
    int8u_t byte2;
    int8u_t A;
    int8u_t X;
    int8u_t Y;
    int8u_t status_register;
    int8u_t stack_offset;
    int16_t vertical_scan;
    int16_t horizontal_scan;
    int16u_t cycle;
};

struct ComputationState {
    int16u_t program_counter;
    int8u_t stack_offset;

    int8u_t A = 0;
    int8u_t X = 0;
    int8u_t Y = 0;

    int8u_t N = 0;
    int8u_t O = 0;
    int8u_t U = 1;
    int8u_t B = 0;
    int8u_t D = 0;
    int8u_t I = 1;
    int8u_t Z = 0;
    int8u_t C = 0;

    int8u_t data1;
    int8u_t data2;

    int8u_t value1;
    int8u_t value2;
    int8u_t value3;

    int16u_t address;

    volatile int8u_t store;

    int8u_t opcode;

    int8u_t controller_read_position = 0;
    int frame_count = 0;

    int16_t vertical_scan = 0;
    int16_t horizontal_scan = 21 - 3;
    uint32_t ppu_cycle = 21 - 3;
    int16u_t instruction_countdown = 1;

    int8u_t ppu_status = 0;
    int8u_t ppu_address_H;
    int8u_t ppu_address_L;
    int16u_t loopy_V;
    int16u_t loopy_T;
    int16u_t loopy_X;
    int16u_t background_sprite_MSB_shift_register;
    int16u_t background_sprite_LSB_shift_register;
    int16u_t background_palette_MSB_shift_register;
    int16u_t background_palette_LSB_shift_register;
    int16u_t nametable_next_address;
    int16u_t background_next_attribute;
    int8u_t character_next_MSB_plane;
    int8u_t character_next_LSB_plane;

    int8u_t fine_X;
    bool ppu_address_latch = false;
    volatile int8u_t ppu_data_buffer;

    bool is_dma_active = false;
    int8u_t dma_count = 0;
    int8u_t dma_source_H;
    int8u_t dma_source_L;

    bool has_blanked = false;
    bool has_vblank_nmi = false;
    int nmi_count = 0;

    __device__
    uint8_t statusRegisterByteGet() {
        return (N << 7) | (O << 6) | (U << 5) | (B << 4) |
               (D << 3) | (I << 2) | (Z << 1) | (C << 0);
    }

};

#define PPU_CTRL1       0x2000
#define PPU_CTRL2       0x2000
#define PPU_STATUS      0x2002
#define PPU_OAM_ADDRESS 0x2003
#define PPU_OAM_DATA    0x2004
#define PPU_SCROLL      0x2005
#define PPU_ADDRESS     0x2006
#define PPU_DATA        0x2007
#define PPU_OAM_DMA     0x4014

#define DMA_read1       0xAB
#define DMA_write1      0xB2

#define CPU_MEMORY         0x0000
#define PPU_REGISTERS      (CPU_MEMORY + 0x800)
#define PPU_MEMORY         (PPU_REGISTERS + 8)
#define CARTRIDGE_MEMORY   (PPU_MEMORY + 0x4000)
#define PPU_OAM_REGISTER   (CARTRIDGE_MEMORY + 0x8000)
#define CONTROL_PORT_1     (PPU_OAM_REGISTER + 1)
#define PPU_OAM_MEMORY     (CONTROL_PORT_1 + 1)
#define NULL_ADDRESS_READ_OFFSET (PPU_OAM_MEMORY + 0x100)
#define NULL_ADDRESS_WRITE_OFFSET (NULL_ADDRESS_READ_OFFSET + 2)

#define NULL_ADDRESS_READ  0x10000
#define NULL_ADDRESS_WRITE (NULL_ADDRESS_READ + 2)

struct Memory {
    int8u_t array[0] = {};
    int8u_t cpu_memory[0x800] = {};
    int8u_t ppu_registers[8] = {};
    int8u_t ppu_memory[0x4000] = {};
    int8u_t cartridge_memory[0x8000] = {};
    int8u_t ppu_OAM_register[1] = {};
    int8u_t control_port1[1] = {};
    int8u_t ppu_OAM_memory[0x100] = {};
    int8u_t null_address_read[2] = {};
    int8u_t null_address_write[2] = {};

    __device__
    int map_offset(int index) {
        return         (CPU_MEMORY)*(0x0000 <= index && index < 0x2000) +
                    (PPU_REGISTERS)*(0x2000 <= index && index < 0x4000) +
        (NULL_ADDRESS_WRITE_OFFSET)*(0x4000 <= index && index < 0x4014) +
                 (PPU_OAM_REGISTER)*(0x4014 == index) +
        (NULL_ADDRESS_WRITE_OFFSET)*(0x4015 == index) +
                   (CONTROL_PORT_1)*(0x4016 == index) +
        (NULL_ADDRESS_WRITE_OFFSET)*(0x4017 <= index && index < 0x8000) +
                 (CARTRIDGE_MEMORY)*(0x8000 <= index && index < 0x10000) +
         (NULL_ADDRESS_READ_OFFSET)*(NULL_ADDRESS_READ == index) +
        (NULL_ADDRESS_WRITE_OFFSET)*(NULL_ADDRESS_WRITE == index);
    }

    __device__
    int map_index(int offset, int index) {
        return        (index % 0x800)*(CPU_MEMORY       == offset) +
               ((index - 0x2000) % 8)*(PPU_REGISTERS    == offset) +
                                  (0)*(PPU_OAM_REGISTER == offset) +
                     (index - 0x8000)*(CARTRIDGE_MEMORY == offset);
    }

    __device__
    int8u_t& ppu_memory_mirrored(int index) {
        return ppu_memory[palette_mirror_address(index)];
    }

    __device__
    int8u_t read(int index0, ComputationState* state) {
        int offset = map_offset(index0);
        int index = map_index(offset, index0);

        /* PPU data pre read */
        bool is_ppu_data_read = (offset == PPU_REGISTERS) && (index == 0x07);
        int16u_t ppu_address = ((state->ppu_address_H << 8) | state->ppu_address_L);
        int16u_t address = offset + index;
        address = (address)*(!is_ppu_data_read) + (PPU_MEMORY + palette_mirror_address(ppu_address))*is_ppu_data_read;

        int8u_t value = array[address];

        /* PPU data buffer */
        bool is_ppu_buffered_read = is_ppu_data_read && (ppu_address < 0x3F00);
        bool is_ppu_immediate_read = is_ppu_data_read && (ppu_address >= 0x3F00);
        int8u_t buffer_temp = state->ppu_data_buffer;
        state->ppu_data_buffer = (state->ppu_data_buffer)*(!is_ppu_data_read) + (value)*(is_ppu_data_read);
        value = (value)*(!is_ppu_data_read) + (value)*is_ppu_immediate_read + (buffer_temp)*is_ppu_buffered_read;

        /* PPU data post read */
        int8u_t ppu_ctrl = ppu_registers[0x00];
        ppu_address += (31*((ppu_ctrl >> 2) & 0x01) + 1)*(is_ppu_data_read);
        state->ppu_address_H = ppu_address >> 8;
        state->ppu_address_L = 0xFF & ppu_address;

        bool is_ppu_status = (offset == PPU_REGISTERS) && (index == 0x02);
        array[offset + index] &= is_ppu_status ? 0b01111111 : 0xFF;
        state->ppu_address_latch = is_ppu_status ? false : state->ppu_address_latch;

        bool is_controller_read = (offset == CONTROL_PORT_1) && (index == 0x00);
        if (is_controller_read)
        {
            value = ((value << state->controller_read_position) & 0x80) >> 7;

            if (state->controller_read_position > 7) {
                value = 0x01;
            }

            state->controller_read_position += 1;
        }

        return value;
    }

    __device__
    int16u_t palette_mirror_address(int16u_t ppu_address) {
        /*
        http://wiki.nesdev.com/w/index.php/PPU_palettes#Memory_Map
        Addresses $3F10/$3F14/$3F18/$3F1C are mirrors of
                  $3F00/$3F04/$3F08/$3F0C */
        bool any_ = (ppu_address == 0x3F10) ||
                    (ppu_address == 0x3F14) ||
                    (ppu_address == 0x3F18) ||
                    (ppu_address == 0x3F1C);

        bool is_nametable_mirror_region1 = (0x2800 <= ppu_address && ppu_address < 0x2C00);
        bool is_nametable_mirror_region2 = (0x2C00 <= ppu_address && ppu_address < 0x3000);

        return (ppu_address)*(!any_) +
               (ppu_address - 0x800)*(is_nametable_mirror_region1) +
               (ppu_address - 0x800)*(is_nametable_mirror_region2) +
               (0x3F00)*(ppu_address == 0x3F10) +
               (0x3F04)*(ppu_address == 0x3F14) +
               (0x3F08)*(ppu_address == 0x3F18) +
               (0x3F0C)*(ppu_address == 0x3F1C);
    }

    __device__
    void write(int index0, int8u_t value, bool is_oam_dma_write, ComputationState* state) {
        int offset = map_offset(index0);
        int index = map_index(offset, index0);

        bool is_ppu_address = (offset == PPU_REGISTERS) && (index == 0x06);
        state->ppu_address_H = (is_ppu_address && state->ppu_address_latch == false) ? value : state->ppu_address_H;
        state->ppu_address_L = (is_ppu_address && state->ppu_address_latch == true) ? value : state->ppu_address_L;

        // /* map to PPU RAM if necessary */

        bool is_ppu_data_write = (offset == PPU_REGISTERS) && (index == 0x07);
        int16u_t ppu_address = ((state->ppu_address_H << 8) | state->ppu_address_L);

        bool is_controller_write = (offset == CONTROL_PORT_1) && (index == 0x00);

        /* PPU data pre write */
        int address = is_ppu_data_write ? PPU_MEMORY + palette_mirror_address(ppu_address) : offset + index;
        address = is_oam_dma_write ? PPU_OAM_MEMORY + index0 : address;
        address = is_controller_write ? NULL_ADDRESS_WRITE_OFFSET : address;

        array[address] = value;

        /* PPU internal state */
        bool is_2000_write = (offset == PPU_REGISTERS) && (index == 0x00);
        bool is_2005_write0 = (offset == PPU_REGISTERS) && (index == 0x05) && (state->ppu_address_latch == false);
        bool is_2005_write1 = (offset == PPU_REGISTERS) && (index == 0x05) && (state->ppu_address_latch == true);
        bool is_2006_write0 = (offset == PPU_REGISTERS) && (index == 0x06) && (state->ppu_address_latch == false);
        bool is_2006_write1 = (offset == PPU_REGISTERS) && (index == 0x06) && (state->ppu_address_latch == true);
        bool any_latch_write = (offset == PPU_REGISTERS) && ((index == 0x05) || (index == 0x06));

        int16u_t value16 = (int16u_t) value;

        state->loopy_T = (state->loopy_T)*(!is_2000_write)  + ((state->loopy_T &  (         ~0b0000000000000011  << 10))
                                                                               | ((value16 & 0b0000000000000011) << 10))*is_2000_write; // nametable set
        state->loopy_T = (state->loopy_T)*(!is_2005_write0) + ((state->loopy_T &  (         ~0b0000000011111000  >>  3))
                                                                               | ((value16 & 0b0000000011111000) >>  3))*is_2005_write0; // x scroll set
        state->loopy_X = (state->loopy_X)*(!is_2005_write0) + (                  ((value16 & 0b0000000000000111) >>  0))*is_2005_write0; // x scroll set
        state->loopy_T = (state->loopy_T)*(!is_2005_write1) + ((state->loopy_T & ~((          0b0000000000000111  << 12) |  (          0b0000000011111000  << 2)))
                                                                               |  ((value16 & 0b0000000000000111) << 12) | ((value16 & 0b0000000011111000) << 2))*is_2005_write1; // y scroll set
        state->loopy_T = (state->loopy_T)*(!is_2006_write0) + (((state->loopy_T &  (         ~0b0000000000111111  <<  8))
                                                                                | ((value16 & 0b0000000000111111) <<  8)) & 0b0011111111111111)*is_2006_write0;
        state->loopy_T = (state->loopy_T)*(!is_2006_write1) + ((state->loopy_T & (           ~0b0000000011111111  <<  0))
                                                                               |  ((value16 & 0b0000000011111111) <<  0))*is_2006_write1;
        state->loopy_V = (state->loopy_V)*(!is_2006_write1) + (state->loopy_T)*is_2006_write1;
        state->ppu_address_latch = (any_latch_write) ? !state->ppu_address_latch : state->ppu_address_latch;


        /* PPU data post write */
        int8u_t ppu_ctrl = ppu_registers[0x00];
        ppu_address += (31*((ppu_ctrl >> 2) & 0x01) + 1)*(is_ppu_data_write);
        state->ppu_address_H = ppu_address >> 8;
        state->ppu_address_L = 0xFF & ppu_address;

        bool is_DMA_register_write = (offset == PPU_OAM_REGISTER) && (index == 0x00);
        state->is_dma_active |= is_DMA_register_write;
        state->dma_source_H = (state->dma_source_H)*(!is_DMA_register_write) + (value)*is_DMA_register_write;
        state->dma_source_L = (state->dma_source_L)*(!is_DMA_register_write) + (0)*is_DMA_register_write;

        // bool is_OAM_address_write = (offset == PPU_REGISTERS) && (index == 0x03);
        // state->dma_target_H = (state->dma_target_H)*(!is_OAM_address_write) + (PPU_OAM_OFFSET >> 8)*is_OAM_address_write;
        // state->dma_target_L = (state->dma_target_L)*(!is_OAM_address_write) + ((PPU_OAM_OFFSET & 0xFF) + value)*is_OAM_address_write;

        state->controller_read_position = (state->controller_read_position)*(!is_controller_write) + (0)*is_controller_write;
    }

    __device__
    int8u_t& operator[](int index) {
        int offset = map_offset(index);
        index = map_index(offset, index);

        return array[offset + index];
    }
};

__device__
const uint8_t colors[256*3] = {
    84, 84, 84, 0, 30, 116, 8, 16, 144, 48, 0, 136, 68, 0, 100, 92, 0, 48, 84, 4, 0, 60, 24, 0, 32, 42, 0, 8, 58, 0, 0, 64, 0, 0, 60, 0, 0, 50, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    152, 150, 152, 8, 76, 196, 48, 50, 236, 92, 30, 228, 136, 20, 176, 160, 20, 100, 152, 34, 32, 120, 60, 0, 84, 90, 0, 40, 114, 0, 8, 124, 0, 0, 118, 40, 0, 102, 120, 0, 0, 0,0, 0, 0, 0, 0, 0,
    236, 238, 236, 76, 154, 236, 120, 124, 236, 176, 98, 236, 228, 84, 236, 236, 88, 180, 236, 106, 100, 212, 136, 32, 160, 170, 0, 116, 196, 0, 76, 208, 32, 56, 204, 108, 56, 180, 204, 60, 60, 60,0, 0, 0, 0, 0, 0,
    236, 238, 236, 168, 204, 236, 188, 188, 236, 212, 178, 236, 236, 174, 236, 236, 174, 212, 236, 180, 176, 228, 196, 144, 204, 210, 120, 180, 222, 120, 168, 226, 144, 152, 226, 180, 160, 214, 228, 160, 162, 160, 0, 0, 0, 0, 0, 0};

struct SystemState {
    int16u_t program_counter_initial;
    Memory memory;
    int8u_t stack_offset_initial;

    uint8_t* frames_red;
    uint8_t* frames_green;
    uint8_t* frames_blue;
    int frames_pixel_index = 0;

    Trace traceLineLast;
    Trace* trace_lines;
    int trace_lines_index = 0;

    SystemState(std::vector<char>& program, int16u_t program_counter, int load_point) {
        // std::copy(program.begin(), program.end(), memory.cartridge_memory);
        std::copy(program.begin(), program.end(), (memory.cartridge_memory + 0x4000));
        // std::copy(character_data.begin(), character_data.end(), memory.ppu_memory);
        this->program_counter_initial = program_counter;
        this->stack_offset_initial = 0xFD;
    }

    SystemState(std::vector<char>& program_data, std::vector<char>& character_data) {
        std::copy(program_data.begin(), program_data.end(), memory.cartridge_memory);
        if (program_data.size() < 0x8000) {
            std::copy(program_data.begin(), program_data.end(), (memory.cartridge_memory + 0x4000));
        }
        std::copy(character_data.begin(), character_data.end(), memory.ppu_memory);
        this->program_counter_initial = (memory.cartridge_memory[0xFFFD % 0x8000] << 8) | memory.cartridge_memory[0xFFFC % 0x8000];
        this->stack_offset_initial = 0xFD;
    }

    __device__
    void traceWrite(uint16_t program_counter, int8u_t* opcodes, ComputationState* state) {
        traceLineLast = {
            .program_counter = program_counter,
            .opcode = opcodes[0],
            .byte1 = opcodes[1],
            .byte2 = opcodes[2],
            .A = state->A,
            .X = state->X,
            .Y = state->Y,
            .status_register = state->statusRegisterByteGet(),
            .stack_offset = state->stack_offset,
            .vertical_scan = state->vertical_scan,
            .horizontal_scan = state->horizontal_scan,
            .cycle = (int16u_t)(state->ppu_cycle / 3)
        };
    }

    __device__
    static void pixelWrite0(SystemState* system, ComputationState* state) {
        if (threadIdx.x == 7 && state->frame_count == MAXFRAMES - 1) {
            int8_t bit1 = (state->background_sprite_MSB_shift_register >> (15 - state->loopy_X)) & 0x01;
            int8_t bit0 = (state->background_sprite_LSB_shift_register >> (15 - state->loopy_X)) & 0x01;
            int8_t color_bits = (bit1 << 1) | bit0;

            if (state->vertical_scan == 0 && state->horizontal_scan == 0) {
                printf("\033c");
            }

            if      (color_bits == 0) { printf("\033[0;37m\u2588"); }
            else if (color_bits == 1) { printf("\033[0;34m\u2588"); }
            else if (color_bits == 2) { printf("\033[0;32m\u2588"); }
            else if (color_bits == 3) { printf("\033[0;31m\u2588"); }
            else {printf("X");}
            if (state->horizontal_scan == 255) {
                printf("\033[0m\n");
            }
        }
    }

    __device__
    static void pixelWrite(SystemState* system, ComputationState* state) {
        if (threadIdx.x == OBSERVED_INSTANCE && state->frame_count == MAXFRAMES - 1) {
            uint8_t bit1 = (state->background_sprite_MSB_shift_register >> (15 - state->loopy_X)) & 0x01;
            uint8_t bit0 = (state->background_sprite_LSB_shift_register >> (15 - state->loopy_X)) & 0x01;
            uint8_t color_bits = (bit1 << 1) | bit0;

            uint8_t palette_bit1 = (state->background_palette_MSB_shift_register >> (15 - state->loopy_X)) & 0x01;
            uint8_t palette_bit0 = (state->background_palette_LSB_shift_register >> (15 - state->loopy_X)) & 0x01;
            uint8_t palette_bits = (palette_bit1 << 1) | palette_bit0;

            system->frames_red[system->frames_pixel_index]   = colors[system->memory.ppu_memory_mirrored(0x3F00 + (palette_bits*4 + color_bits))*3 + 0];
            system->frames_green[system->frames_pixel_index] = colors[system->memory.ppu_memory_mirrored(0x3F00 + (palette_bits*4 + color_bits))*3 + 1];
            system->frames_blue[system->frames_pixel_index]  = colors[system->memory.ppu_memory_mirrored(0x3F00 + (palette_bits*4 + color_bits))*3 + 2];

            system->frames_pixel_index++;
        }
    }

    __device__
    static void scanLineNext(SystemState* system, ComputationState* state) {
        state->horizontal_scan = (state->horizontal_scan + 1) % 341;
        state->vertical_scan += (state->horizontal_scan == 0);
        state->vertical_scan = (state->vertical_scan == 261) ? -1 : state->vertical_scan;

        if ((state->horizontal_scan == 340) && (state->frame_count & 1) && (state->vertical_scan == -1)) {
            state->vertical_scan = 0;
            state->horizontal_scan = 0;
        }

        bool blank_condition = !(state->has_blanked) &&
                                (state->vertical_scan == 241) &&
                                (state->horizontal_scan == 1);

        system->memory[PPU_STATUS] |= blank_condition ? 0x80 : 0x00;
        state->has_blanked = state->has_blanked || blank_condition;
        state->has_blanked = state->has_blanked && (state->vertical_scan > 0);
        state->has_vblank_nmi = state->has_vblank_nmi && (state->vertical_scan > 0);

        bool is_vblank_clear = (state->vertical_scan == -1) && (state->horizontal_scan == 1);
        system->memory[PPU_STATUS] &= (0xFF)*(!is_vblank_clear) + (~0x80)*is_vblank_clear;

        state->frame_count += (state->vertical_scan == -1) && (state->horizontal_scan == 0);

        bool is_rendering_enabled = (0 != ((system->memory[0x2001] >> 3) & 0x03));
        bool is_background_enabled = (0 != ((system->memory[0x2001] >> 3) & 0x01));

        /* x increment */
        bool is_coarse_X_increment = (state->horizontal_scan == 328) || (state->horizontal_scan == 336) || ((state->horizontal_scan != 0 && state->horizontal_scan < 256) && (state->horizontal_scan % 8 == 0));
        is_coarse_X_increment &= is_rendering_enabled;
        uint8_t coarse_X = (state->loopy_V & 0b0000000000011111);
        coarse_X = (coarse_X + 1) & 0x1F;
        int16u_t new_loopy_V = (state->loopy_V & ~0b0000000000011111) | coarse_X;
        new_loopy_V = (new_loopy_V)*(coarse_X != 0) + (new_loopy_V ^ 0b0000010000000000)*(coarse_X == 0);
        state->loopy_V = (state->loopy_V)*(!is_coarse_X_increment) + (new_loopy_V)*(is_coarse_X_increment);

        /* y increment */
        bool is_Y_increment = (state->horizontal_scan == 256) && is_rendering_enabled;
        uint16_t fine_Y = (state->loopy_V & 0b0111000000000000) >> 12;
        uint16_t coarse_Y = (state->loopy_V & 0b0000001111100000) >> 5;
        fine_Y = (fine_Y + 1) & 0x07;
        coarse_Y += (fine_Y == 0x00);
        new_loopy_V = (state->loopy_V & ~0b0111001111100000) | (fine_Y << 12) | (coarse_Y << 5);
        state->loopy_V = (state->loopy_V)*(!is_Y_increment) + (new_loopy_V)*is_Y_increment;

        bool is_horizontal_prepare = (state->horizontal_scan == 257) && is_rendering_enabled;
        new_loopy_V = (state->loopy_V & ~0b0000010000011111) | (state->loopy_T & 0b0000010000011111);
        state->loopy_V = (state->loopy_V)*(!is_horizontal_prepare) + (new_loopy_V)*is_horizontal_prepare;

        bool is_vertical_prepare = (state->vertical_scan == -1) && (state->horizontal_scan == 280) && is_rendering_enabled;
        new_loopy_V = (state->loopy_V & ~0b0111101111100000) | (state->loopy_T & 0b0111101111100000);
        state->loopy_V = (state->loopy_V)*(!is_vertical_prepare) + (new_loopy_V)*is_vertical_prepare;

        state->fine_X = (state->fine_X + 1) & 0x07;

        /* https://wiki.nesdev.com/w/images/d/d1/Ntsc_timing.png */
        if ((state->vertical_scan < 240) && (1 <= state->horizontal_scan && state->horizontal_scan <= 257) || (321 <= state->horizontal_scan)) {
            state->background_sprite_MSB_shift_register <<= is_background_enabled;
            state->background_sprite_LSB_shift_register <<= is_background_enabled;
            state->background_palette_MSB_shift_register <<= is_background_enabled;
            state->background_palette_LSB_shift_register <<= is_background_enabled;

            int16u_t address = 0;

            switch ((state->horizontal_scan - 1) % 8) {
                case 0:
                    // load shift register
                    state->background_sprite_MSB_shift_register = (state->background_sprite_MSB_shift_register & 0xFF00) | state->character_next_MSB_plane;
                    state->background_sprite_LSB_shift_register = (state->background_sprite_LSB_shift_register & 0xFF00) | state->character_next_LSB_plane;
                    state->background_palette_MSB_shift_register = (state->background_palette_MSB_shift_register & 0xFF00) | ((state->background_next_attribute >> 1) & 0x01)*0x00FF;
                    state->background_palette_LSB_shift_register = (state->background_palette_LSB_shift_register & 0xFF00) | ((state->background_next_attribute >> 0) & 0x01)*0x00FF;
                    break;
                case 1:
                    // nametable byte
                    state->nametable_next_address = system->memory.ppu_memory_mirrored(0x2000 | (state->loopy_V & 0x0FFF));
                    break;
                case 2: break;
                case 3:
                    // attribute byte
                    struct {
                        uint16_t _0[0];
                        uint16_t coarse_X : 5;
                        uint16_t coarse_Y : 5;
                        uint16_t N0 : 1;
                        uint16_t N1 : 1;
                        uint16_t fine_Y : 3;
                        uint16_t unused : 1;
                    } V;

                    V._0[0] = state->loopy_V;

                    address = 0x23C0 | (state->loopy_V & 0x0C00) | ((state->loopy_V >> 4) & 0x38) | ((state->loopy_V >> 2) & 0x07); 
                    state->background_next_attribute = system->memory.ppu_memory_mirrored(address);

                    if (V.coarse_Y & 0x02) {
                        state->background_next_attribute >>= 4;
                    }

                    if (V.coarse_X & 0x02) {
                        state->background_next_attribute >>= 2;
                    }

                    state->background_next_attribute &= 0x03;

                    break;

                case 4: break;
                case 5:
                    // background LSB bit plane
                    address = ((((uint16_t)system->memory[PPU_CTRL1] >> 4) & 0x01) << 12) +
                              ((uint16_t)state->nametable_next_address << 4) +
                              ((state->loopy_V & 0b0111000000000000) >> 12);

                    state->character_next_LSB_plane = system->memory.ppu_memory_mirrored(address);
                    break;
                case 6: break;
                case 7:
                    // background MSB bit plane
                    address = ((((uint16_t)system->memory[PPU_CTRL1] >> 4) & 0x01) << 12) +
                              ((uint16_t)state->nametable_next_address << 4) +
                              ((state->loopy_V & 0b0111000000000000) >> 12);

                    state->character_next_MSB_plane = system->memory.ppu_memory_mirrored(address + 8);
                    break;
            }
        }

        if (0 <= state->vertical_scan && state->vertical_scan < 240 && state->horizontal_scan < 256) {
            pixelWrite(system, state);
        }
    }

    int stack_pointer = 0;
    int count = 0;
    int count1 = 0;

    __device__
    void next(ComputationState* state) {
        int8u_t* opcodes = &memory[state->program_counter];

        int8u_t opcode = opcodes[0];
        state->data1 = opcodes[1];
        state->data2 = opcodes[2];

        state->ppu_cycle++;
        scanLineNext(this, state);
        state->ppu_cycle++;
        scanLineNext(this, state);
        state->ppu_cycle++;
        scanLineNext(this, state);

        state->instruction_countdown -= (!state->is_dma_active);
        bool instruction_OK = state->instruction_countdown == 0;
        opcode = (0x6B)*(!instruction_OK) + (opcode)*instruction_OK;

        bool nmi_condition = instruction_OK &&
                             (!state->has_vblank_nmi) &&
                             ((memory[PPU_CTRL1] & 0x80) == 0x80) && // NMI enabled
                             ((memory[PPU_STATUS] & 0x80) == 0x80); // vblank has occurred

        state->has_vblank_nmi |= nmi_condition;
        state->nmi_count += nmi_condition;

        int8u_t nmi_L = memory[0xFFFA];
        int8u_t nmi_H = memory[0xFFFB];

        opcode = (opcode)*(!nmi_condition) + (0xC2)*(nmi_condition);
        state->data1 = (state->data1)*(!nmi_condition) + (nmi_L)*nmi_condition;
        state->data2 = (state->data2)*(!nmi_condition) + (nmi_H)*nmi_condition;

        bool odd_cycle = ((state->is_dma_active) && (state->ppu_cycle % 2 == 1));
        bool even_cycle = ((state->is_dma_active) && (state->ppu_cycle % 2 == 0));
        opcode = (opcode)*(!odd_cycle) + (DMA_read1)*odd_cycle;
        opcode = (opcode)*(!even_cycle) + (DMA_write1)*even_cycle;

        state->data1 = (state->data1)*(!odd_cycle) + (state->dma_source_L + state->dma_count)*odd_cycle;
        state->data2 = (state->data2)*(!odd_cycle) + (state->dma_source_H)*odd_cycle;
        state->data1 = (state->data1)*(!even_cycle) + (state->dma_count)*even_cycle;

        state->dma_count += even_cycle;
        state->is_dma_active = (state->is_dma_active && !(even_cycle && state->dma_count == 0));

        count++;

        #ifdef DEBUG
        if (threadIdx.x == 0 && instruction_OK) {
            traceWrite(state->program_counter, opcodes, state);
            trace_lines[trace_lines_index] = traceLineLast;
            trace_lines_index++;
        }
        #endif

        state->opcode = opcode;
        operationTransition(opcode, this, state);
    }
};

struct OperationInformation {
    std::string name = "NOP"; 
    int byte_count = 0;
    std::string format_type = "None";

    std::string doFormat(int8u_t byte1, int8u_t byte2, int16u_t program_counter) {
        char cs[10];

        if (format_type == "Absolute" || format_type == "AbsoluteDereference") {
            sprintf(cs, "$%04X ", (((uint8_t)byte2 << 8) | (uint8_t)byte1) & 0xFFFF);
        } else if (format_type == "AbsoluteAddressDereference") {
            sprintf(cs, "($%04X) ", (((uint8_t)byte2 << 8) | (uint8_t)byte1) & 0xFFFF);
        } else if (format_type == "AbsoluteX") {
            sprintf(cs, "$%04X,X ", (((uint8_t)byte2 << 8) | (uint8_t)byte1) & 0xFFFF);
        } else if (format_type == "AbsoluteY") {
            sprintf(cs, "$%04X,Y ", (((uint8_t)byte2 << 8) | (uint8_t)byte1) & 0xFFFF);
        } else if (format_type == "Immediate") {
            sprintf(cs, "#$%02X ", (uint8_t)byte1);
        } else if (format_type == "Zeropage" || format_type == "ZeropageDereference") {
            sprintf(cs, "$%02X ", (uint8_t)byte1);
        } else if (format_type == "IndirectX") {
            sprintf(cs, "($%02X,X) ", (uint8_t)byte1);
        } else if (format_type == "ZeropageX") {
            sprintf(cs, "$%02X,X ", (uint8_t)byte1);
        } else if (format_type == "ZeropageY") {
            sprintf(cs, "$%02X,Y ", (uint8_t)byte1);
        } else if (format_type == "IndirectY") {
            sprintf(cs, "($%02X),Y ", (uint8_t)byte1);
        } else if (format_type == "Implied") {
            strcpy(cs, "");
        } else if (format_type == "Address_Relative") {
            sprintf(cs, "$%04X ", program_counter + 2 + (int8_t)byte1);
        } else {
            return format_type;
        }

        return std::string(cs);
    }
};
