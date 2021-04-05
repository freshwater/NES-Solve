
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

    int8u_t opcode;
    int8u_t data1;
    int8u_t data2;

    int8u_t value1;
    int8u_t value2;
    int8u_t value3;

    int16u_t address;

    volatile int8u_t store;
    int8u_t ppu_status;

    int8u_t controller_read_position = 0;
    int frame_count = 0;
    int num_actions = 0;

    int16_t vertical_scan = 0;
    int16_t horizontal_scan = 21 - 3;
    uint32_t ppu_cycle = 21 - 3;
    int16u_t instruction_countdown = 1;

    struct Sprite {
        int8u_t Y;
        int8u_t tile_id;
        int8u_t attributes;
        int8u_t X;
    };

    Sprite sprites[64];
    bool is_OAM_memory_invalidated = true;

    uint32_t sprite_MSB_line[9];
    uint32_t sprite_LSB_line[9];
    uint32_t sprite_palette_bit1_line[9];
    uint32_t sprite_palette_bit0_line[9];
    uint32_t sprite_zero_line[9];
    bool has_sprite_zero_hit;
    int sprites_intersecting_index = 0;

    int8u_t ppu_address_H;
    int8u_t ppu_address_L;
    int16u_t loopy_V;
    int16u_t loopy_T;
    int16u_t loopy_X;
    int16u_t background_sprite_MSB_shift_register;
    int16u_t background_sprite_LSB_shift_register;
    int16u_t background_palette_MSB_shift_register;
    int16u_t background_palette_LSB_shift_register;
    int16u_t nametable_next_tile_id;
    int16u_t background_next_attribute;
    int8u_t character_next_MSB_plane;
    int8u_t character_next_LSB_plane;

    bool ppu_address_latch = false;
    volatile int8u_t ppu_data_buffer;

    bool is_DMA_should_start = false;
    bool is_DMA_active = false;
    int8u_t DMA_index = 0;
    int8u_t DMA_source_H;
    int8u_t DMA_source_L;

    bool has_blanked = false;
    bool has_vblank_nmi = false;
    int nmi_count = 0;

    uint8_t control_port1;

    __device__
    uint8_t statusRegisterByteGet() {
        return (N << 7) | (O << 6) | (U << 5) | (B << 4) |
               (D << 3) | (I << 2) | (Z << 1) | (C << 0);
    }

};

#define PPU_CTRL    0x2000
#define PPU_STATUS  0x2002

#define NOP_instruction         0x6B
#define NMI_instruction         0xC2
#define DMA_read1_instruction   0xAB
#define DMA_write1_instruction  0xB2

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
    int mapOffset(int index) {
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
    int mapIndex(int offset, int index) {
        return        (index % 0x800)*(CPU_MEMORY       == offset) +
               ((index - 0x2000) % 8)*(PPU_REGISTERS    == offset) +
                                  (0)*(PPU_OAM_REGISTER == offset) +
                     (index - 0x8000)*(CARTRIDGE_MEMORY == offset);
    }

    __device__
    __inline__
    int8u_t& ppuMemoryMapped(int index) {
        return ppu_memory[ppuMapAddress(index)];
    }

    __device__
    int8u_t read(int index0, uint8_t* program_data, ComputationState* state) {
        int offset = mapOffset(index0);
        int index = mapIndex(offset, index0);

        /* PPU data pre read */
        bool is_ppu_data_read = (offset == PPU_REGISTERS) && (index == 0x07);
        int16u_t ppu_address = ((state->ppu_address_H << 8) | state->ppu_address_L);
        int16u_t address = offset + index;
        address = (address)*(!is_ppu_data_read) + (PPU_MEMORY + ppuMapAddress(ppu_address))*is_ppu_data_read;

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

        bool is_2002_read = (offset == PPU_REGISTERS) && (index == 0x02);
        value = (value)*(!is_2002_read) + (state->ppu_status)*is_2002_read;
        state->ppu_status &= is_2002_read ? 0b01111111 : 0xFF;
        state->ppu_address_latch = is_2002_read ? false : state->ppu_address_latch;

        bool is_controller_read = (offset == CONTROL_PORT_1) && (index == 0x00);
        if (is_controller_read)
        {
            value = state->control_port1;
            value = ((value << state->controller_read_position) & 0x80) >> 7;

            if (state->controller_read_position > 7) {
                value = 0x01;
            }

            state->controller_read_position += 1;
        }

        return value;
    }

    __device__
    __inline__
    int16u_t ppuMapAddress(int16u_t address) {
        bool is_palette_address = (0x3F00 <= address);
        address = (address)*(!is_palette_address) + (0x3F00 | (address & 0x001F))*is_palette_address;

        /*
        http://wiki.nesdev.com/w/index.php/PPU_palettes#Memory_Map
        Addresses $3F10/$3F14/$3F18/$3F1C are mirrors of $3F00/$3F04/$3F08/$3F0C */
        /* The underlying memory on these appears to be used only in rare/hacky situations.
           For simplicity I've consolodated them all to $3F00 */

        bool any_palette_background = (0x3F00 < address && ((address & 0x03) == 0));
        bool is_nametable_mirror_region1 = (0x2800 <= address && address < 0x2C00);
        bool is_nametable_mirror_region2 = (0x2C00 <= address && address < 0x3000);

        address = (address)*(!is_nametable_mirror_region1) + (address - 0x800)*(is_nametable_mirror_region1);
        address = (address)*(!is_nametable_mirror_region2) + (address - 0x800)*(is_nametable_mirror_region2);

        auto value =  (address)*(!any_palette_background) +
                       (0x3F00)*(any_palette_background);

        return value;
    }

    __device__
    void write(int index0, int8u_t value, bool is_oam_dma_write, ComputationState* state) {
        int offset = mapOffset(index0);
        int index = mapIndex(offset, index0);

        bool is_ppu_address = (offset == PPU_REGISTERS) && (index == 0x06);
        state->ppu_address_H = (is_ppu_address && state->ppu_address_latch == false) ? value : state->ppu_address_H;
        state->ppu_address_L = (is_ppu_address && state->ppu_address_latch == true) ? value : state->ppu_address_L;

        // /* map to PPU RAM if necessary */

        bool is_ppu_data_write = (offset == PPU_REGISTERS) && (index == 0x07);
        int16u_t ppu_address = ((state->ppu_address_H << 8) | state->ppu_address_L);

        bool is_controller_write = (offset == CONTROL_PORT_1) && (index == 0x00);

        /* PPU data pre write */
        int address = is_ppu_data_write ? PPU_MEMORY + ppuMapAddress(ppu_address) : offset + index;
        address = is_oam_dma_write ? PPU_OAM_MEMORY + index0 : address;
        address = is_controller_write ? NULL_ADDRESS_WRITE_OFFSET : address;

        array[address] = value;

        /* PPU internal state */
        bool is_2000_write = (offset == PPU_REGISTERS) && (index == 0x00);
        bool is_2002_write = (offset == PPU_REGISTERS) && (index == 0x02);
        bool is_2005_write0 = (offset == PPU_REGISTERS) && (index == 0x05) && (state->ppu_address_latch == false);
        bool is_2005_write1 = (offset == PPU_REGISTERS) && (index == 0x05) && (state->ppu_address_latch == true);
        bool is_2006_write0 = (offset == PPU_REGISTERS) && (index == 0x06) && (state->ppu_address_latch == false);
        bool is_2006_write1 = (offset == PPU_REGISTERS) && (index == 0x06) && (state->ppu_address_latch == true);
        bool any_latch_write = (offset == PPU_REGISTERS) && ((index == 0x05) || (index == 0x06));

        state->ppu_status = (state->ppu_status)*(!is_2002_write) + value*is_2002_write;

        int16u_t value16 = (int16u_t) value;
        state->loopy_T = (state->loopy_T)*(!is_2000_write)  + ((state->loopy_T &   (         ~0b00000000'00000011  << 10))
                                                                               |  ((value16 & 0b00000000'00000011) << 10))*is_2000_write; // nametable set
        state->loopy_T = (state->loopy_T)*(!is_2005_write0) + ((state->loopy_T &   (         ~0b00000000'11111000  >>  3))
                                                                               |  ((value16 & 0b00000000'11111000) >>  3))*is_2005_write0; // x scroll set
        state->loopy_X = (state->loopy_X)*(!is_2005_write0) + (                   ((value16 & 0b00000000'00000111) >>  0))*is_2005_write0;
        state->loopy_T = (state->loopy_T)*(!is_2005_write1) + ((state->loopy_T & ~((          0b00000000'00000111  << 12) |  (          0b00000000'11111000  << 2)))
                                                                               |  ((value16 & 0b00000000'00000111) << 12) | ((value16 & 0b00000000'11111000) << 2))*is_2005_write1; // y scroll set
        state->loopy_T = (state->loopy_T)*(!is_2006_write0) + (((state->loopy_T &  (         ~0b00000000'00111111  <<  8))
                                                                                | ((value16 & 0b00000000'00111111) <<  8)) & 0b00111111'11111111)*is_2006_write0;
        state->loopy_T = (state->loopy_T)*(!is_2006_write1) + ((state->loopy_T &   (         ~0b00000000'11111111  <<  0))
                                                                               |  ((value16 & 0b00000000'11111111) <<  0))*is_2006_write1;
        state->loopy_V = (state->loopy_V)*(!is_2006_write1) + (state->loopy_T)*is_2006_write1;
        state->ppu_address_latch = (any_latch_write) ? !state->ppu_address_latch : state->ppu_address_latch;

        /* PPU data post write */
        int8u_t ppu_ctrl = ppu_registers[0x00];
        ppu_address += (31*((ppu_ctrl >> 2) & 0x01) + 1)*(is_ppu_data_write);
        state->ppu_address_H = ppu_address >> 8;
        state->ppu_address_L = 0xFF & ppu_address;

        bool is_DMA_register_write = (offset == PPU_OAM_REGISTER) && (index == 0x00);
        state->is_DMA_should_start = (state->is_DMA_should_start || is_DMA_register_write);
        state->DMA_source_H = (state->DMA_source_H)*(!is_DMA_register_write) + (value)*is_DMA_register_write;
        state->DMA_source_L = (state->DMA_source_L)*(!is_DMA_register_write) + (0)*is_DMA_register_write;

        // bool is_OAM_address_write = (offset == PPU_REGISTERS) && (index == 0x03);
        // state->dma_target_H = (state->dma_target_H)*(!is_OAM_address_write) + (PPU_OAM_OFFSET >> 8)*is_OAM_address_write;
        // state->dma_target_L = (state->dma_target_L)*(!is_OAM_address_write) + ((PPU_OAM_OFFSET & 0xFF) + value)*is_OAM_address_write;

        state->controller_read_position = (state->controller_read_position)*(!is_controller_write) + (0)*is_controller_write;
    }

    __device__
    int8u_t& operator[](int index) {
        int offset = mapOffset(index);
        index = mapIndex(offset, index);

        return array[offset + index];
    }
};

__device__
__constant__
const uint8_t colors[256*3] = {
    84, 84, 84, 0, 30, 116, 8, 16, 144, 48, 0, 136, 68, 0, 100, 92, 0, 48, 84, 4, 0, 60, 24, 0, 32, 42, 0, 8, 58, 0, 0, 64, 0, 0, 60, 0, 0, 50, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    152, 150, 152, 8, 76, 196, 48, 50, 236, 92, 30, 228, 136, 20, 176, 160, 20, 100, 152, 34, 32, 120, 60, 0, 84, 90, 0, 40, 114, 0, 8, 124, 0, 0, 118, 40, 0, 102, 120, 0, 0, 0,0, 0, 0, 0, 0, 0,
    236, 238, 236, 76, 154, 236, 120, 124, 236, 176, 98, 236, 228, 84, 236, 236, 88, 180, 236, 106, 100, 212, 136, 32, 160, 170, 0, 116, 196, 0, 76, 208, 32, 56, 204, 108, 56, 180, 204, 60, 60, 60,0, 0, 0, 0, 0, 0,
    236, 238, 236, 168, 204, 236, 188, 188, 236, 212, 178, 236, 236, 174, 236, 236, 174, 212, 236, 180, 176, 228, 196, 144, 204, 210, 120, 180, 222, 120, 168, 226, 144, 152, 226, 180, 160, 214, 228, 160, 162, 160, 0, 0, 0, 0, 0, 0};

__device__
void operationTransition(uint8_t, SystemState*, ComputationState*, Memory&);

struct SystemState {
    int16u_t program_counter_initial;
    Memory global_memory;
    int8u_t stack_offset_initial;

    int8u_t* program_data;

    uint8_t* frames_red;
    uint8_t* frames_green;
    uint8_t* frames_blue;
    int frames_pixel_index = 0;

    Trace traceLineLast;
    Trace* trace_lines;
    int trace_lines_index = 0;

    SystemState(std::vector<char>& program, int16u_t program_counter, int load_point) {
        std::copy(program.begin(), program.end(), global_memory.cartridge_memory);
        std::copy(program.begin(), program.end(), (global_memory.cartridge_memory + 0x4000));
        this->program_counter_initial = program_counter;
        this->stack_offset_initial = 0xFD;
    }

    SystemState(std::vector<char>& program_data, std::vector<char>& character_data) {
        std::copy(program_data.begin(), program_data.end(), global_memory.cartridge_memory);
        if (program_data.size() < 0x8000) {
            std::copy(program_data.begin(), program_data.end(), (global_memory.cartridge_memory + 0x4000));
        }
        std::copy(character_data.begin(), character_data.end(), global_memory.ppu_memory);
        this->program_counter_initial = (global_memory.cartridge_memory[0xFFFD % 0x8000] << 8) | global_memory.cartridge_memory[0xFFFC % 0x8000];
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
    static void pixelWrite(SystemState* system, ComputationState* state, Memory& memory) {
        uint8_t bit1 = (state->background_sprite_MSB_shift_register >> (15 - state->loopy_X)) & 0x01;
        uint8_t bit0 = (state->background_sprite_LSB_shift_register >> (15 - state->loopy_X)) & 0x01;
        uint8_t color_bits = (bit1 << 1) | bit0;

        uint8_t palette_bit1 = (state->background_palette_MSB_shift_register >> (15 - state->loopy_X)) & 0x01;
        uint8_t palette_bit0 = (state->background_palette_LSB_shift_register >> (15 - state->loopy_X)) & 0x01;
        uint8_t palette_bits = (palette_bit1 << 1) | palette_bit0;

        uint8_t sprite_bit1 =                  (state->sprite_MSB_line[state->horizontal_scan / 32] >> (31 - (state->horizontal_scan % 32))) & 0x01;
        uint8_t sprite_bit0 =                  (state->sprite_LSB_line[state->horizontal_scan / 32] >> (31 - (state->horizontal_scan % 32))) & 0x01;
        uint8_t sprite_palette_bit1 = (state->sprite_palette_bit1_line[state->horizontal_scan / 32] >> (31 - (state->horizontal_scan % 32))) & 0x01;
        uint8_t sprite_palette_bit0 = (state->sprite_palette_bit0_line[state->horizontal_scan / 32] >> (31 - (state->horizontal_scan % 32))) & 0x01;
        uint8_t sprite_color_bits = (sprite_bit1 << 1) | sprite_bit0;
        uint8_t sprite_palette_bits = (sprite_palette_bit1 << 1) | sprite_palette_bit0;

        palette_bits = (palette_bits)*(sprite_color_bits == 0) + (sprite_palette_bits)*(sprite_color_bits != 0);
        color_bits = (color_bits)*(sprite_color_bits == 0) + (sprite_color_bits);

        if (state->frame_count == state->num_actions - 1) {
            system->frames_red[system->frames_pixel_index*(gridDim.x*blockDim.x) + blockIdx.x*(blockDim.x) + threadIdx.x]   = colors[memory.ppuMemoryMapped(0x3F00 + 0x10*(sprite_color_bits != 0) + (palette_bits*4 + color_bits))*3 + 0];
            system->frames_green[system->frames_pixel_index*(gridDim.x*blockDim.x) + blockIdx.x*(blockDim.x) + threadIdx.x] = colors[memory.ppuMemoryMapped(0x3F00 + 0x10*(sprite_color_bits != 0) + (palette_bits*4 + color_bits))*3 + 1];
            system->frames_blue[system->frames_pixel_index*(gridDim.x*blockDim.x) + blockIdx.x*(blockDim.x) + threadIdx.x]  = colors[memory.ppuMemoryMapped(0x3F00 + 0x10*(sprite_color_bits != 0) + (palette_bits*4 + color_bits))*3 + 2];

            system->frames_pixel_index++;
        }

        if (!state->has_sprite_zero_hit) {
            uint8_t sprite_zero_bit = (state->sprite_zero_line[state->horizontal_scan / 32] >> (31 - (state->horizontal_scan % 32))) & 0x01;
            if (sprite_zero_bit != 0 && color_bits != 0) {
                state->ppu_status |= 0x40;
                state->has_sprite_zero_hit = true;
            }
        }
    }

    __device__
    static void scanlineNext(SystemState* system, ComputationState* state, Memory& memory) {
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

        state->ppu_status |= blank_condition ? 0x80 : 0x00;

        state->has_blanked = state->has_blanked || blank_condition;
        state->has_blanked = state->has_blanked && (state->vertical_scan > 0);
        state->has_vblank_nmi = state->has_vblank_nmi && (state->vertical_scan > 0);

        if ((state->vertical_scan == -1) && (state->horizontal_scan == 1)) {
            state->ppu_status &= ~0xC0;
            state->has_sprite_zero_hit = false;
        }

        state->frame_count += (state->vertical_scan == -1) && (state->horizontal_scan == 0);

        bool is_rendering_enabled = true; //  (0 != ((system->memory[0x2001] >> 3) & 0x03));
        bool is_background_enabled = true; // (0 != ((system->memory[0x2001] >> 3) & 0x01));

        bool is_vertical_prepare = (state->vertical_scan == -1) && (state->horizontal_scan == 280) && is_rendering_enabled;
        int16u_t new_loopy_V = (state->loopy_V & ~0b0111101111100000) | (state->loopy_T & 0b0111101111100000);
        state->loopy_V = (state->loopy_V)*(!is_vertical_prepare) + (new_loopy_V)*is_vertical_prepare;

        if (state->horizontal_scan == 257 && 0 <= state->vertical_scan && state->vertical_scan < 240) {
            if (state->is_OAM_memory_invalidated) {
                memcpy(state->sprites, memory.ppu_OAM_memory, 256);
                state->is_OAM_memory_invalidated = false;
            }

            state->sprites_intersecting_index = 0;

            for (int i = 0; i < 9; i++) {
                state->sprite_MSB_line[i] = 0;
                state->sprite_LSB_line[i] = 0;
                state->sprite_zero_line[i] = 0;
            }

            uint8_t sprite_MSB_planes[8];
            uint8_t sprite_LSB_planes[8];
            uint8_t sprite_palette_bit1s[8];
            uint8_t sprite_palette_bit0s[8];
            uint8_t sprite_Xs[8];
            bool has_sprite_zero = false;

            for (int i = 0; i < 64; i++) {
                if ((state->sprites_intersecting_index < 8) && (0 <= state->vertical_scan - state->sprites[i].Y) && (state->vertical_scan - state->sprites[i].Y < 8)) {
                    uint16_t address = // ((((uint16_t)system->memory[PPU_CTRL] >> 3) & 0x01) << 12) +
                                         ((uint16_t)state->sprites[i].tile_id << 4) +
                                          (state->vertical_scan - state->sprites[i].Y);
                                         // ((state->loopy_V & 0b0111000000000000) >> 12);

                    bool is_flip_horizontally = (state->sprites[i].attributes >> 6) & 0x01;

                    sprite_MSB_planes[state->sprites_intersecting_index] = memory.ppuMemoryMapped(address + 8);
                    sprite_LSB_planes[state->sprites_intersecting_index] = memory.ppuMemoryMapped(address);
                    uint8_t MSB_reversed = (uint8_t)(__brev((uint32_t) sprite_MSB_planes[state->sprites_intersecting_index]) >> (32 - 8));
                    uint8_t LSB_reversed = (uint8_t)(__brev((uint32_t) sprite_LSB_planes[state->sprites_intersecting_index]) >> (32 - 8));
                    sprite_MSB_planes[state->sprites_intersecting_index] = (sprite_MSB_planes[state->sprites_intersecting_index])*(!is_flip_horizontally) + (MSB_reversed)*is_flip_horizontally;
                    sprite_LSB_planes[state->sprites_intersecting_index] = (sprite_LSB_planes[state->sprites_intersecting_index])*(!is_flip_horizontally) + (LSB_reversed)*is_flip_horizontally;

                    sprite_palette_bit1s[state->sprites_intersecting_index] = 0xFF*((state->sprites[i].attributes >> 1) & 0x01);
                    sprite_palette_bit0s[state->sprites_intersecting_index] = 0xFF*((state->sprites[i].attributes >> 0) & 0x01);
                    sprite_Xs[state->sprites_intersecting_index] = state->sprites[i].X;

                    state->sprites_intersecting_index++;

                    if (i == 0) {
                        has_sprite_zero = true;
                    }
                }
            }

            for (int i = state->sprites_intersecting_index - 1; i > -1; i--) {
                int32_t MSB_segment1 = state->sprite_MSB_line[sprite_Xs[i] / 32];
                int32_t LSB_segment1 = state->sprite_LSB_line[sprite_Xs[i] / 32];
                int32_t palette_bit1_segment1 = state->sprite_palette_bit1_line[sprite_Xs[i] / 32];
                int32_t palette_bit0_segment1 = state->sprite_palette_bit0_line[sprite_Xs[i] / 32];

                int32_t MSB_segment2 = state->sprite_MSB_line[sprite_Xs[i] / 32 + 1];
                int32_t LSB_segment2 = state->sprite_LSB_line[sprite_Xs[i] / 32 + 1];
                int32_t palette_bit1_segment2 = state->sprite_palette_bit1_line[sprite_Xs[i] / 32 + 1];
                int32_t palette_bit0_segment2 = state->sprite_palette_bit0_line[sprite_Xs[i] / 32 + 1];

                MSB_segment1 &=          ~(((uint32_t)                   0xFF << (32 - 8)) >> (sprite_Xs[i] % 32));
                MSB_segment1 |=           (((uint32_t)   sprite_MSB_planes[i] << (32 - 8)) >> (sprite_Xs[i] % 32));
                LSB_segment1 &=          ~(((uint32_t)                   0xFF << (32 - 8)) >> (sprite_Xs[i] % 32));
                LSB_segment1 |=           (((uint32_t)   sprite_LSB_planes[i] << (32 - 8)) >> (sprite_Xs[i] % 32));
                palette_bit1_segment1 &= ~(((uint32_t)                   0xFF << (32 - 8)) >> (sprite_Xs[i] % 32));
                palette_bit1_segment1 |=  (((uint32_t)sprite_palette_bit1s[i] << (32 - 8)) >> (sprite_Xs[i] % 32));
                palette_bit0_segment1 &= ~(((uint32_t)                   0xFF << (32 - 8)) >> (sprite_Xs[i] % 32));
                palette_bit0_segment1 |=  (((uint32_t)sprite_palette_bit0s[i] << (32 - 8)) >> (sprite_Xs[i] % 32));

                MSB_segment2 &=          ~(((uint32_t)                   0xFF << (32 - 8 + (32 - (sprite_Xs[i] % 32)))));
                MSB_segment2 |=           (((uint32_t)   sprite_MSB_planes[i] << (32 - 8 + (32 - (sprite_Xs[i] % 32)))));
                LSB_segment2 &=          ~(((uint32_t)                   0xFF << (32 - 8 + (32 - (sprite_Xs[i] % 32)))));
                LSB_segment2 |=           (((uint32_t)   sprite_LSB_planes[i] << (32 - 8 + (32 - (sprite_Xs[i] % 32)))));
                palette_bit1_segment2 &= ~(((uint32_t)                   0xFF << (32 - 8 + (32 - (sprite_Xs[i] % 32)))));
                palette_bit1_segment2 |=  (((uint32_t)sprite_palette_bit1s[i] << (32 - 8 + (32 - (sprite_Xs[i] % 32)))));
                palette_bit0_segment2 &= ~(((uint32_t)                   0xFF << (32 - 8 + (32 - (sprite_Xs[i] % 32)))));
                palette_bit0_segment2 |=  (((uint32_t)sprite_palette_bit0s[i] << (32 - 8 + (32 - (sprite_Xs[i] % 32)))));

                state->sprite_MSB_line[sprite_Xs[i] / 32] = MSB_segment1;
                state->sprite_LSB_line[sprite_Xs[i] / 32] = LSB_segment1;
                state->sprite_palette_bit1_line[sprite_Xs[i] / 32] = palette_bit1_segment1;
                state->sprite_palette_bit0_line[sprite_Xs[i] / 32] = palette_bit0_segment1;
                state->sprite_MSB_line[(sprite_Xs[i] / 32) + 1] = MSB_segment2;
                state->sprite_LSB_line[(sprite_Xs[i] / 32) + 1] = LSB_segment2;
                state->sprite_palette_bit1_line[sprite_Xs[i] / 32 + 1] = palette_bit1_segment2;
                state->sprite_palette_bit0_line[sprite_Xs[i] / 32 + 1] = palette_bit0_segment2;
            }

            if (has_sprite_zero) {
                int32_t sprite_zero_segment1 = state->sprite_zero_line[sprite_Xs[0] / 32];
                int32_t sprite_zero_segment2 = state->sprite_zero_line[sprite_Xs[0] / 32 + 1];
                sprite_zero_segment1 &= ~(((uint32_t)                                          0xFF << (32 - 8)) >> (sprite_Xs[0] % 32));
                sprite_zero_segment1 |=  (((uint32_t) (sprite_MSB_planes[0] | sprite_LSB_planes[0]) << (32 - 8)) >> (sprite_Xs[0] % 32));
                sprite_zero_segment2 &= ~(((uint32_t)                                          0xFF << (32 - 8 + (32 - (sprite_Xs[0] % 32)))));
                sprite_zero_segment2 |=  (((uint32_t) (sprite_MSB_planes[0] | sprite_LSB_planes[0]) << (32 - 8 + (32 - (sprite_Xs[0] % 32)))));
                state->sprite_zero_line[sprite_Xs[0] / 32] = sprite_zero_segment1;
                state->sprite_zero_line[sprite_Xs[0] / 32 + 1] = sprite_zero_segment2;
            }
        }

        /* https://wiki.nesdev.com/w/images/d/d1/Ntsc_timing.png */
        if ((state->vertical_scan < 240) && (1 <= state->horizontal_scan && state->horizontal_scan <= 257) || (321 <= state->horizontal_scan && state->horizontal_scan <= 336)) {
            /* Y increment */
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

            state->background_sprite_MSB_shift_register <<= is_background_enabled;
            state->background_sprite_LSB_shift_register <<= is_background_enabled;
            state->background_palette_MSB_shift_register <<= is_background_enabled;
            state->background_palette_LSB_shift_register <<= is_background_enabled;

            int16u_t address = 0;

            switch (state->horizontal_scan % 8) {
                case 0:
                    // load shift register
                    state->background_sprite_MSB_shift_register = (state->background_sprite_MSB_shift_register & 0xFF00) | state->character_next_MSB_plane;
                    state->background_sprite_LSB_shift_register = (state->background_sprite_LSB_shift_register & 0xFF00) | state->character_next_LSB_plane;
                    state->background_palette_MSB_shift_register = (state->background_palette_MSB_shift_register & 0xFF00) | ((state->background_next_attribute >> 1) & 0x01)*0x00FF;
                    state->background_palette_LSB_shift_register = (state->background_palette_LSB_shift_register & 0xFF00) | ((state->background_next_attribute >> 0) & 0x01)*0x00FF;
                    break;
                case 1:
                    // nametable byte
                    state->nametable_next_tile_id = memory.ppuMemoryMapped(0x2000 | (state->loopy_V & 0x0FFF));
                    break;
                case 2: break;
                case 3:
                    // attribute byte
                    address = 0x23C0 | (state->loopy_V & 0b00001100'00000000) | ((state->loopy_V & 0b00000011'10000000) >> 4) | ((state->loopy_V & 0b00000000'00011100) >> 2);
                    state->background_next_attribute = memory.ppuMemoryMapped(address);

                    state->background_next_attribute >>= 4*((state->loopy_V & 0b0'000'00'00010'00000) != 0);
                    state->background_next_attribute >>= 2*((state->loopy_V & 0b0'000'00'00000'00010) != 0);
                    state->background_next_attribute &= 0x03;
                    break;
                case 4: break;
                case 5:
                    // background LSB bit plane
                    address = ((((uint16_t)memory.ppu_registers[0x00] >> 4) & 0x01) << 12) +
                              ((uint16_t)state->nametable_next_tile_id << 4) +
                              ((state->loopy_V & 0b0111000000000000) >> 12);

                    state->character_next_LSB_plane = memory.ppuMemoryMapped(address);
                    break;
                case 6: break;
                case 7:
                    // background MSB bit plane
                    address = ((((uint16_t)memory.ppu_registers[0x00] >> 4) & 0x01) << 12) +
                              ((uint16_t)state->nametable_next_tile_id << 4) +
                              ((state->loopy_V & 0b0111000000000000) >> 12);

                    state->character_next_MSB_plane = memory.ppuMemoryMapped(address + 8);

                    /* X increment */
                    uint8_t coarse_X = (state->loopy_V & 0b0000000000011111);
                    coarse_X = (coarse_X + 1) & 0x1F;
                    int16u_t new_loopy_V = (state->loopy_V & ~0b0000000000011111) | coarse_X;
                    state->loopy_V = (new_loopy_V)*(coarse_X != 0) + (new_loopy_V ^ 0b0000010000000000)*(coarse_X == 0);
                    break;
            }
        }

        if (0 <= state->vertical_scan && state->vertical_scan < 240 && state->horizontal_scan < 256) {
            pixelWrite(system, state, memory);
        }
    }

    __device__
    void next(ComputationState* state, Memory& memory) {
        int8u_t opcode = memory[state->program_counter + 0];
        state->data1   = memory[state->program_counter + 1];
        state->data2   = memory[state->program_counter + 2];

        state->ppu_cycle++;
        scanlineNext(this, state, memory);
        state->ppu_cycle++;
        scanlineNext(this, state, memory);
        state->ppu_cycle++;
        scanlineNext(this, state, memory);

        state->is_DMA_active = (state->is_DMA_active || (state->is_DMA_should_start && state->ppu_cycle % 2 == 1));

        // clear is_DMA_should_start as soon as is_DMA_active is set
        state->is_DMA_should_start = (state->is_DMA_should_start != (state->is_DMA_should_start && state->ppu_cycle % 2 == 1));

        state->instruction_countdown -= (!state->is_DMA_active);
        bool instruction_OK = state->instruction_countdown == 0;
        opcode = (NOP_instruction)*(!instruction_OK) + (opcode)*instruction_OK;

        bool nmi_condition = instruction_OK &&
                             (!state->has_vblank_nmi) &&
                             ((memory.ppu_registers[0x00] & 0x80) == 0x80) && // NMI enabled
                             ((state->ppu_status & 0x80) == 0x80); // vblank has occurred

        state->has_vblank_nmi |= nmi_condition;
        state->nmi_count += nmi_condition;

        int8u_t nmi_L = memory[0xFFFA];
        int8u_t nmi_H = memory[0xFFFB];

        opcode = (opcode)*(!nmi_condition) + (NMI_instruction)*(nmi_condition);
        state->data1 = (state->data1)*(!nmi_condition) + (nmi_L)*nmi_condition;
        state->data2 = (state->data2)*(!nmi_condition) + (nmi_H)*nmi_condition;

        bool odd_cycle = ((state->is_DMA_active) && (state->ppu_cycle % 2 == 1));
        bool even_cycle = ((state->is_DMA_active) && (state->ppu_cycle % 2 == 0));
        opcode = (opcode)*(!odd_cycle) + (DMA_read1_instruction)*odd_cycle;
        opcode = (opcode)*(!even_cycle) + (DMA_write1_instruction)*even_cycle;

        state->data1 = (state->data1)*(!odd_cycle) + (state->DMA_source_L + state->DMA_index)*odd_cycle;
        state->data2 = (state->data2)*(!odd_cycle) + (state->DMA_source_H)*odd_cycle;
        state->data1 = (state->data1)*(!even_cycle) + (state->DMA_index)*even_cycle;

        state->DMA_index += even_cycle;
        bool new_is_DMA_active = (state->is_DMA_active && !(even_cycle && state->DMA_index == 0));
        state->is_OAM_memory_invalidated = (state->is_OAM_memory_invalidated) || (new_is_DMA_active != state->is_DMA_active);
        state->is_DMA_active = new_is_DMA_active;

        #ifdef DEBUG
        if (instruction_OK) {
            uint8_t* opcodes = &memory[state->program_counter];
            traceWrite(state->program_counter, opcodes, state);
            trace_lines[trace_lines_index] = traceLineLast;
            trace_lines_index++;
        }
        #endif

        state->opcode = opcode;
        operationTransition(opcode, this, state, memory);
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
