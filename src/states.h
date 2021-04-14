
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

    int8u_t store;
    int8u_t ppu_status;

    int controller_read_position = 0;

    int vertical_scan = -1;
    int horizontal_scan = 21 - 3;
    int instruction_countdown = 1;

    int64_t ppu_cycle = 21 - 3;
    int64_t frame_count = 0;
    int64_t num_actions = 0;

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
    bool sprite_zero_hit_possible;
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

    uint8_t control_port1;

    float kernel_sums[256];
    uint8_t kernel_sums_index = 0;
    int kernel_X = 0;
    int kernel_Y = 0;
    float kernel_sum = 0;

    __device__
    uint8_t statusRegisterByteGet() {
        return (N << 7) | (O << 6) | (U << 5) | (B << 4) |
               (D << 3) | (I << 2) | (Z << 1) | (C << 0);
    }

};

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
#define CONTROL_PORT_2     (CONTROL_PORT_1 + 1)
#define PPU_OAM_MEMORY     (CONTROL_PORT_2 + 1)
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
    int8u_t control_port2[1] = {};
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
                   (CONTROL_PORT_2)*(0x4017 == index) +
        (NULL_ADDRESS_WRITE_OFFSET)*(0x4018 <= index && index < 0x8000) +
                 (CARTRIDGE_MEMORY)*(0x8000 <= index && index < 0x10000) +
         (NULL_ADDRESS_READ_OFFSET)*(NULL_ADDRESS_READ == index) +
        (NULL_ADDRESS_WRITE_OFFSET)*(NULL_ADDRESS_WRITE == index);
    }

    __device__
    int mapIndex(int offset, int index) {
        return    (index & 0x7FF)*(CPU_MEMORY       == offset) +
           ((index - 0x2000) & 7)*(PPU_REGISTERS    == offset) +
                              (0)*(PPU_OAM_REGISTER == offset) +
                              (0)*(CONTROL_PORT_1   == offset) +
                              (0)*(CONTROL_PORT_2   == offset) +
                              (0)*(PPU_OAM_REGISTER == offset) +
                 (index - 0x8000)*(CARTRIDGE_MEMORY == offset) +
                              (0)*(NULL_ADDRESS_READ_OFFSET  == offset) +
                              (0)*(NULL_ADDRESS_WRITE_OFFSET == offset);
    }

    __device__
    int8u_t& ppuMemoryMapped(int index) {
        return ppu_memory[ppuMapAddress(index)];
    }

    __device__
    int8u_t ppuPaletteRead(int index) {
        index &= 0x1F;
        bool any_palette_background = ((index & 0x03) == 0);
        index = any_palette_background ? 0 : index;

        return ppu_memory[0x3F00 + index];
    }

    __device__
    int16u_t ppuMapAddress(int16u_t address) {
        bool is_palette_address = (0x3F00 <= address);
        address = is_palette_address ? (0x3F00 | (address & 0x001F)) : address;

        /*
        http://wiki.nesdev.com/w/index.php/PPU_palettes#Memory_Map
        Addresses $3F00/$3F04/$3F08/$3F0C are mirrored by $3F10/$3F14/$3F18/$3F1C. */
        /* Aside from the mirroring, the read/write works as normal for the CPU,
           but the PPU render circuit reads these from $3F00 (see ppuPaletteRead). */

        bool any_palette_mirror = (0x3F10 <= address && ((address & 0x03) == 0));
        bool is_nametable_mirror_region1 = (0x2800 <= address && address < 0x2C00);
        bool is_nametable_mirror_region2 = (0x2C00 <= address && address < 0x3000);

        address = is_nametable_mirror_region1 ? (address - 0x800) : address;
        address = is_nametable_mirror_region2 ? (address - 0x800) : address;

        address = any_palette_mirror ? (address - 0x0010) : address;

        return address;
    }

    __device__
    int8u_t read(int index0, ComputationState* state) {
        int offset = mapOffset(index0);
        int index = mapIndex(offset, index0);
        int16u_t address = offset + index;

        bool is_ppu_data_read = (offset == PPU_REGISTERS) && (index == 0x07);
        int16u_t ppu_address = ((state->ppu_address_H << 8) | state->ppu_address_L);
        address = is_ppu_data_read ? (PPU_MEMORY + ppuMapAddress(ppu_address)) : address;

        int value = array[address];

        if (is_ppu_data_read) {
            bool is_ppu_buffered_read = ppu_address < 0x3F00;
            int8u_t buffer_temp = state->ppu_data_buffer;
            state->ppu_data_buffer = value;
            value = is_ppu_buffered_read ? buffer_temp : value;

            int8u_t ppu_ctrl = ppu_registers[0x00];
            ppu_address += (31*((ppu_ctrl >> 2) & 0x01) + 1);
            state->ppu_address_H = ppu_address >> 8;
            state->ppu_address_L = 0xFF & ppu_address;
        }

        if ((offset == PPU_REGISTERS) && (index == 0x02)) {
            value = state->ppu_status;
            state->ppu_status &= 0b01111111;
            state->ppu_address_latch = false;
        }

        if (offset == CONTROL_PORT_1)
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

        /* https://wiki.nesdev.com/w/index.php/PPU_scrolling#Register_controls */
        state->loopy_T = is_2000_write ?   ((state->loopy_T &  (       ~(0b00000000'00000011  << 10)))
                                                            |  ((value & 0b00000000'00000011) << 10)) : state->loopy_T; // nametable set
        state->loopy_T = is_2005_write0 ?  ((state->loopy_T &  (       ~(0b00000000'11111000  >>  3)))
                                                            |  ((value & 0b00000000'11111000) >>  3)) : state->loopy_T; // x scroll set
        state->loopy_X = is_2005_write0 ?  (                   ((value & 0b00000000'00000111) >>  0)) : state->loopy_X;
        state->loopy_T = is_2005_write1 ?  ((state->loopy_T & ~((        0b00000000'00000111  << 12) | (         0b00000000'11111000  << 2)))
                                                            |  ((value & 0b00000000'00000111) << 12) | ((value & 0b00000000'11111000) << 2)) : state->loopy_T; // y scroll set
        state->loopy_T = is_2006_write0 ? (((state->loopy_T &  (       ~(0b00000000'00111111  <<  8)))
                                                            |  ((value & 0b00000000'00111111) <<  8)) & 0b00111111'11111111) : state->loopy_T;
        state->loopy_T = is_2006_write1 ?  ((state->loopy_T &  (       ~(0b00000000'11111111  <<  0)))
                                                            |  ((value & 0b00000000'11111111) <<  0)) : state->loopy_T;
        state->loopy_V = is_2006_write1 ? (state->loopy_T) : state->loopy_V;
        state->ppu_address_latch = any_latch_write ? (!state->ppu_address_latch) : state->ppu_address_latch;

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

        if (is_controller_write) {
            state->controller_read_position = 0;
        }
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
const float kernel[16*16*3] = {
    0.125     , 0.12598039, 0.12696589, 0.12795651, 0.12895228, 0.1299532 , 0.13095928, 0.13197055, 0.13298701, 0.13400867, 0.13503555, 0.13606766, 0.13710502, 0.13814764, 0.13919554,
    0.14024871, 0.14130719, 0.14237098, 0.14344009, 0.14451455, 0.14559435, 0.14667952, 0.14777007, 0.14886601, 0.14996736, 0.15107413, 0.15218632, 0.15330397, 0.15442707, 0.15555564,
    0.1566897 , 0.15782926, 0.15897433, 0.16012492, 0.16128105, 0.16244274, 0.16360998, 0.16478281, 0.16596123, 0.16714525, 0.16833489, 0.16953017, 0.17073109, 0.17193766, 0.17314991,
    0.17436784, 0.17559147, 0.17682081, 0.17805588, 0.17929668, 0.18054323, 0.18179555, 0.18305365, 0.18431753, 0.18558723, 0.18686274, 0.18814408, 0.18943126, 0.1907243 , 0.19202321,
    0.19332801, 0.1946387 , 0.19595531, 0.19727784, 0.1986063 , 0.19994072, 0.2012811 , 0.20262746, 0.2039798 , 0.20533816, 0.20670253, 0.20807292, 0.20944937, 0.21083187, 0.21222044,
    0.21361509, 0.21501584, 0.2164227 , 0.21783568, 0.2192548 , 0.22068008, 0.22211151, 0.22354912, 0.22499292, 0.22644292, 0.22789914, 0.22936159, 0.23083029, 0.23230524, 0.23378646,
    0.23527396, 0.23676776, 0.23826786, 0.2397743 , 0.24128706, 0.24280618, 0.24433166, 0.24586351, 0.24740176, 0.24894641, 0.25049747, 0.25205496, 0.2536189 , 0.25518929, 0.25676615,
    0.25834949, 0.25993933, 0.26153568, 0.26313855, 0.26474795, 0.26636391, 0.26798643, 0.26961552, 0.2712512 , 0.27289348, 0.27454238, 0.27619791, 0.27786008, 0.2795289 , 0.2812044 ,
    0.28288657, 0.28457544, 0.28627102, 0.28797332, 0.28968236, 0.29139814, 0.29312069, 0.29485001, 0.29658612, 0.29832903, 0.30007876, 0.30183531, 0.30359871, 0.30536896, 0.30714608,
    0.30893008, 0.31072097, 0.31251877, 0.3143235 , 0.31613515, 0.31795376, 0.31977933, 0.32161187, 0.3234514 , 0.32529793, 0.32715148, 0.32901205, 0.33087966, 0.33275433, 0.33463606,
    0.33652488, 0.33842079, 0.34032381, 0.34223395, 0.34415122, 0.34607564, 0.34800722, 0.34994598, 0.35189192, 0.35384506, 0.35580542, 0.357773  , 0.35974782, 0.3617299 , 0.36371924,
    0.36571587, 0.36771979, 0.36973101, 0.37174956, 0.37377543, 0.37580866, 0.37784925, 0.3798972 , 0.38195255, 0.3840153 , 0.38608545, 0.38816304, 0.39024807, 0.39234054, 0.39444049,
    0.39654791, 0.39866283, 0.40078526, 0.4029152 , 0.40505268, 0.4071977 , 0.40935028, 0.41151044, 0.41367818, 0.41585352, 0.41803647, 0.42022705, 0.42242527, 0.42463114, 0.42684468,
    0.4290659 , 0.43129481, 0.43353142, 0.43577576, 0.43802782, 0.44028764, 0.44255521, 0.44483055, 0.44711368, 0.44940461, 0.45170335, 0.45400991, 0.45632432, 0.45864658, 0.4609767 ,
    0.4633147 , 0.46566059, 0.46801439, 0.47037611, 0.47274575, 0.47512335, 0.4775089 , 0.47990242, 0.48230393, 0.48471344, 0.48713096, 0.4895565 , 0.49199009, 0.49443172, 0.49688142,
    0.49933919, 0.50180506, 0.50427903, 0.50676113, 0.50925135, 0.51174971, 0.51425624, 0.51677094, 0.51929382, 0.52182489, 0.52436418, 0.5269117 , 0.52946745, 0.53203145, 0.53460372,
    0.53718426, 0.5397731 , 0.54237024, 0.5449757 , 0.54758948, 0.55021162, 0.5528421 , 0.55548096, 0.55812821, 0.56078385, 0.5634479 , 0.56612038, 0.56880129, 0.57149065, 0.57418848,
    0.57689478, 0.57960958, 0.58233288, 0.58506469, 0.58780504, 0.59055393, 0.59331137, 0.59607739, 0.59885199, 0.60163518, 0.60442699, 0.60722742, 0.61003649, 0.6128542 , 0.61568058,
    0.61851564, 0.62135938, 0.62421183, 0.627073  , 0.62994289, 0.63282153, 0.63570892, 0.63860509, 0.64151003, 0.64442378, 0.64734633, 0.6502777 , 0.65321792, 0.65616698, 0.6591249 ,
    0.6620917 , 0.66506739, 0.66805198, 0.67104549, 0.67404792, 0.6770593 , 0.68007964, 0.68310894, 0.68614722, 0.68919451, 0.6922508 , 0.69531611, 0.69839046, 0.70147385, 0.70456631,
    0.70766785, 0.71077847, 0.7138982 , 0.71702704, 0.72016501, 0.72331212, 0.72646839, 0.72963382, 0.73280844, 0.73599225, 0.73918527, 0.74238751, 0.74559899, 0.74881972, 0.7520497 ,
    0.75528896, 0.75853752, 0.76179537, 0.76506253, 0.76833903, 0.77162486, 0.77492005, 0.77822461, 0.78153855, 0.78486189, 0.78819463, 0.79153679, 0.79488839, 0.79824944, 0.80161994,
    0.80499993, 0.8083894 , 0.81178836, 0.81519685, 0.81861486, 0.82204241, 0.82547952, 0.82892619, 0.83238245, 0.8358483 , 0.83932375, 0.84280883, 0.84630354, 0.8498079 , 0.85332192,
    0.85684561, 0.86037899, 0.86392207, 0.86747486, 0.87103738, 0.87460965, 0.87819166, 0.88178344, 0.885385  , 0.88899636, 0.89261752, 0.8962485 , 0.89988932, 0.90353998, 0.9072005 ,
    0.9108709 , 0.91455118, 0.91824136, 0.92194146, 0.92565148, 0.92937144, 0.93310135, 0.93684123, 0.9405911 , 0.94435095, 0.94812081, 0.95190069, 0.9556906 , 0.95949056, 0.96330058,
    0.96712067, 0.97095084, 0.97479112, 0.97864151, 0.98250202, 0.98637267, 0.99025348, 0.99414445, 0.9980456 , 1.00195695, 1.0058785 , 1.00981026, 1.01375226, 1.01770451, 1.02166702,
    1.02563979, 1.02962286, 1.03361622, 1.03761989, 1.04163389, 1.04565823, 1.04969292, 1.05373797, 1.05779341, 1.06185923, 1.06593546, 1.07002211, 1.07411919, 1.07822672, 1.0823447 ,
    1.08647316, 1.0906121 , 1.09476154, 1.09892149, 1.10309196, 1.10727298, 1.11146454, 1.11566667, 1.11987938, 1.12410268, 1.12833658, 1.13258111, 1.13683626, 1.14110206, 1.14537851,
    1.14966564, 1.15396345, 1.15827196, 1.16259118, 1.16692112, 1.1712618 , 1.17561324, 1.17997543, 1.18434841, 1.18873217, 1.19312674, 1.19753213, 1.20194835, 1.20637541, 1.21081333,
    1.21526212, 1.21972179, 1.22419236, 1.22867384, 1.23316624, 1.23766959, 1.24218388, 1.24670913, 1.25124537, 1.25579259, 1.26035081, 1.26492006, 1.26950033, 1.27409165, 1.27869402,
    1.28330747, 1.28793199, 1.29256762, 1.29721435, 1.30187221, 1.3065412 , 1.31122134, 1.31591265, 1.32061513, 1.3253288 , 1.33005368, 1.33478977, 1.33953709, 1.34429566, 1.34906548,
    1.35384657, 1.35863894, 1.36344261, 1.36825759, 1.37308389, 1.37792153, 1.38277051, 1.38763086, 1.39250258, 1.3973857 , 1.40228022, 1.40718615, 1.41210351, 1.41703231, 1.42197257,
    1.4269243 , 1.43188751, 1.43686222, 1.44184843, 1.44684617, 1.45185544, 1.45687626, 1.46190865, 1.4669526 , 1.47200815, 1.4770753 , 1.48215407, 1.48724446, 1.49234649, 1.49746018,
    1.50258554, 1.50772258, 1.51287132, 1.51803176, 1.52320393, 1.52838783, 1.53358347, 1.53879088, 1.54401007, 1.54924104, 1.55448381, 1.5597384 , 1.56500482, 1.57028308, 1.57557319,
    1.58087517, 1.58618903, 1.59151479, 1.59685245, 1.60220204, 1.60756356, 1.61293702, 1.61832245, 1.62371986, 1.62912924, 1.63455064, 1.63998404, 1.64542947, 1.65088695, 1.65635647,
    1.66183807, 1.66733174, 1.67283751, 1.67835539, 1.68388538, 1.68942752, 1.69498179, 1.70054823, 1.70612684, 1.71171764, 1.71732064, 1.72293585, 1.72856329, 1.73420297, 1.73985491,
    1.74551911, 1.75119559, 1.75688436, 1.76258544, 1.76829884, 1.77402458, 1.77976266, 1.7855131 , 1.79127591, 1.79705111, 1.80283871, 1.80863872, 1.81445116, 1.82027604, 1.82611337,
    1.83196317, 1.83782545, 1.84370022, 1.8495875 , 1.85548729, 1.86139962, 1.8673245 , 1.87326193, 1.87921194, 1.88517453, 1.89114972, 1.89713753, 1.90313796, 1.90915103, 1.91517675,
    1.92121514, 1.92726621, 1.93332997, 1.93940643, 1.94549561, 1.95159753, 1.95771219, 1.96383961, 1.9699798 , 1.97613278, 1.98229856, 1.98847715, 1.99466856, 2.00087281, 2.00708992,
    2.01331989, 2.01956273, 2.02581847, 2.03208712, 2.03836868, 2.04466318, 2.05097062, 2.05729101, 2.06362438, 2.06997074, 2.07633009, 2.08270246, 2.08908784, 2.09548627, 2.10189775,
    2.10832229, 2.11475991, 2.12121063, 2.12767444, 2.13415138, 2.14064144, 2.14714465, 2.15366102, 2.16019056, 2.16673328, 2.1732892 , 2.17985834, 2.18644069, 2.19303628, 2.19964513,
    2.20626724, 2.21290262, 2.2195513 , 2.22621328, 2.23288858, 2.23957721, 2.24627918, 2.25299451, 2.25972321, 2.26646529, 2.27322077, 2.27998966, 2.28677198, 2.29356773, 2.30037693,
    2.3071996 , 2.31403574, 2.32088537, 2.32774851, 2.33462516, 2.34151534, 2.34841907, 2.35533635, 2.36226721, 2.36921164, 2.37616968, 2.38314132, 2.39012658, 2.39712549, 2.40413804,
    2.41116425, 2.41820414, 2.42525772, 2.43232501, 2.43940601, 2.44650074, 2.45360921, 2.46073144, 2.46786744, 2.47501722, 2.48218079, 2.48935818, 2.49654939, 2.50375443, 2.51097333,
    2.51820608, 2.52545271, 2.53271323, 2.53998766, 2.547276  , 2.55457826, 2.56189447, 2.56922464, 2.57656877, 2.58392689, 2.591299  , 2.59868512, 2.60608527, 2.61349945, 2.62092767,
    2.62836996, 2.63582633, 2.64329678, 2.65078133, 2.65828   , 2.6657928 , 2.67331974, 2.68086084, 2.6884161 , 2.69598554, 2.70356918, 2.71116703, 2.7187791 , 2.72640541, 2.73404596,
    2.74170077, 2.74936986, 2.75705324, 2.76475092, 2.77246291, 2.78018923, 2.78792989, 2.79568491, 2.80345429, 2.81123806, 2.81903622, 2.82684878, 2.83467577, 2.84251719, 2.85037306,
    2.85824339, 2.8661282 , 2.87402749, 2.88194128, 2.88986959, 2.89781243, 2.9057698 , 2.91374173, 2.92172823, 2.9297293 , 2.93774497, 2.94577525, 2.95382015, 2.96187968, 2.96995386,
    2.9780427 , 2.98614621, 2.99426441, 3.00239731, 3.01054493, 3.01870727, 3.02688435, 3.03507618, 3.04328278, 3.05150417, 3.05974034, 3.06799132, 3.07625712, 3.08453775, 3.09283324,
    3.10114358, 3.10946879, 3.11780889, 3.12616389, 3.1345338 , 3.14291864, 3.15131842, 3.15973315, 3.16816284, 3.17660752, 3.18506719, 3.19354187, 3.20203156, 3.21053629, 3.21905606,
    3.22759089, 3.2361408 , 3.24470579, 3.25328588, 3.26188108, 3.27049141, 3.27911688, 3.2877575 , 3.29641328, 3.30508425, 3.31377041, 3.32247177, 3.33118835, 3.33992016, 3.34866722,
    3.35742954, 3.36620713, 3.375};

__device__
void operationTransition(uint8_t, SystemState*, ComputationState*, Memory&);

struct SystemState {
    int16u_t program_counter_initial;
    Memory global_memory;
    int8u_t stack_offset_initial;

    uint8_t* frames_red;
    uint8_t* frames_green;
    uint8_t* frames_blue;
    int64_t frames_pixel_index = 0;
    float* data_lines;
    int64_t data_lines_index = 0;

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
        int background_palette_bit1 = (state->background_palette_MSB_shift_register >> (15 - state->loopy_X)) & 0x01;
        int background_palette_bit0 = (state->background_palette_LSB_shift_register >> (15 - state->loopy_X)) & 0x01;
        int background_palette_bits = (background_palette_bit1 << 1) | background_palette_bit0;

        int background_color_bit1 = (state->background_sprite_MSB_shift_register >> (15 - state->loopy_X)) & 0x01;
        int background_color_bit0 = (state->background_sprite_LSB_shift_register >> (15 - state->loopy_X)) & 0x01;
        int background_color_bits = (background_color_bit1 << 1) | background_color_bit0;

        int sprite_palette_bit1 = (state->sprite_palette_bit1_line[state->horizontal_scan / 32] >> (31 - (state->horizontal_scan % 32))) & 0x01;
        int sprite_palette_bit0 = (state->sprite_palette_bit0_line[state->horizontal_scan / 32] >> (31 - (state->horizontal_scan % 32))) & 0x01;
        int sprite_palette_bits = (sprite_palette_bit1 << 1) | sprite_palette_bit0;

        int sprite_color_bit1 = (state->sprite_MSB_line[state->horizontal_scan / 32] >> (31 - (state->horizontal_scan % 32))) & 0x01;
        int sprite_color_bit0 = (state->sprite_LSB_line[state->horizontal_scan / 32] >> (31 - (state->horizontal_scan % 32))) & 0x01;
        int sprite_color_bits = (sprite_color_bit1 << 1) | sprite_color_bit0;

        int palette_bits = (background_palette_bits)*(sprite_color_bits == 0) + (sprite_palette_bits)*(sprite_color_bits != 0);
        int color_bits = (background_color_bits)*(sprite_color_bits == 0) + (sprite_color_bits);

        int color_offset = memory.ppuPaletteRead(0x10*(sprite_color_bits != 0) + (palette_bits*4 + color_bits))*3;

        if (state->frame_count == state->num_actions - 1) {
            int64_t instance_pixel_index = system->frames_pixel_index*(gridDim.x*blockDim.x) + blockIdx.x*(blockDim.x) + threadIdx.x;
            system->frames_red[instance_pixel_index]   = colors[color_offset + 0];
            system->frames_green[instance_pixel_index] = colors[color_offset + 1];
            system->frames_blue[instance_pixel_index]  = colors[color_offset + 2];

            system->frames_pixel_index++;
        }

        float red = colors[color_offset + 0];
        float green = colors[color_offset + 1];
        float blue = colors[color_offset + 2];

        state->kernel_sum += red*kernel[3*16*state->kernel_Y + 3*state->kernel_X + 0] +
                             green*kernel[3*16*state->kernel_Y + 3*state->kernel_X + 1] +
                             blue*kernel[3*16*state->kernel_Y + 3*state->kernel_X + 2];

        state->kernel_X = (state->kernel_X + 1) & 0xF;

        if (state->kernel_X == 0) {
            state->kernel_sums[state->kernel_sums_index] = state->kernel_sum;
            state->kernel_sums_index++;

            state->kernel_sum = 0;

            if (state->horizontal_scan == 255) {
                state->kernel_Y = (state->kernel_Y + 1) & 0xF;

                if (state->kernel_Y == 0) {
                    for (int i = 0; i < 16; i++) {
                        float kernel_total = state->kernel_sums[i+16*0] + state->kernel_sums[i+16*1] +
                                             state->kernel_sums[i+16*2] + state->kernel_sums[i+16*3] +
                                             state->kernel_sums[i+16*4] + state->kernel_sums[i+16*5] +
                                             state->kernel_sums[i+16*6] + state->kernel_sums[i+16*7] +
                                             state->kernel_sums[i+16*8] + state->kernel_sums[i+16*9] +
                                             state->kernel_sums[i+16*10] + state->kernel_sums[i+16*11] +
                                             state->kernel_sums[i+16*12] + state->kernel_sums[i+16*13] +
                                             state->kernel_sums[i+16*14] + state->kernel_sums[i+16*15];

                        int64_t instance_data_index = system->data_lines_index*(gridDim.x*blockDim.x) + blockIdx.x*(blockDim.x) + threadIdx.x;
                        system->data_lines[instance_data_index] = kernel_total / 16 / 16 / 3 / 255;
                        system->data_lines_index++;
                    }
                }
            }
        }

        if (state->sprite_zero_hit_possible && !state->has_sprite_zero_hit) {
            int sprite_zero_bit = (state->sprite_zero_line[state->horizontal_scan / 32] >> (31 - (state->horizontal_scan % 32))) & 0x01;
            if (sprite_zero_bit != 0 && background_color_bits != 0) {
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

        bool is_rendering_enabled = (memory.ppu_registers[0x01] & 0b0001'1000) != 0;
        bool is_background_enabled = (memory.ppu_registers[0x01] & 0b0000'1000) != 0;

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
                    uint16_t address = (((memory.ppu_registers[0x00] >> 3) & 0x01) << 12) +
                                        ((uint16_t)state->sprites[i].tile_id << 4) +
                                         (state->vertical_scan - state->sprites[i].Y);

                    bool is_flip_horizontally = (state->sprites[i].attributes >> 6) & 0x01;

                    sprite_MSB_planes[state->sprites_intersecting_index] = memory.ppuMemoryMapped(address + 8);
                    sprite_LSB_planes[state->sprites_intersecting_index] = memory.ppuMemoryMapped(address);

                    if (is_flip_horizontally) {
                        uint8_t MSB_reversed = (uint8_t)(__brev((uint32_t) sprite_MSB_planes[state->sprites_intersecting_index]) >> (32 - 8));
                        uint8_t LSB_reversed = (uint8_t)(__brev((uint32_t) sprite_LSB_planes[state->sprites_intersecting_index]) >> (32 - 8));
                        sprite_MSB_planes[state->sprites_intersecting_index] = MSB_reversed;
                        sprite_LSB_planes[state->sprites_intersecting_index] = LSB_reversed;
                    }

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

                state->sprite_zero_hit_possible = true;
            } else {
                state->sprite_zero_hit_possible = false;
            }
        }

        /* https://wiki.nesdev.com/w/images/d/d1/Ntsc_timing.png */
        if ((state->vertical_scan < 240) && (1 <= state->horizontal_scan && state->horizontal_scan <= 257) || (321 <= state->horizontal_scan && state->horizontal_scan <= 336)) {
            if (is_background_enabled) {
                state->background_sprite_MSB_shift_register <<= 1;
                state->background_sprite_LSB_shift_register <<= 1;
                state->background_palette_MSB_shift_register <<= 1;
                state->background_palette_LSB_shift_register <<= 1;
            }

            int16u_t address = 0;

            switch (state->horizontal_scan % 8) {
                case 0:
                    /* Y increment */
                    if ((state->horizontal_scan == 256) && is_rendering_enabled) {
                        uint16_t fine_Y = (state->loopy_V & 0b0111000000000000) >> 12;
                        uint16_t coarse_Y = (state->loopy_V & 0b0000001111100000) >> 5;
                        fine_Y = (fine_Y + 1) & 0x07;
                        coarse_Y += (fine_Y == 0x00);
                        state->loopy_V = (state->loopy_V & ~0b0111001111100000) | (fine_Y << 12) | (coarse_Y << 5);
                    }

                    // load shift register
                    state->background_sprite_MSB_shift_register = (state->background_sprite_MSB_shift_register & 0xFF00) | state->character_next_MSB_plane;
                    state->background_sprite_LSB_shift_register = (state->background_sprite_LSB_shift_register & 0xFF00) | state->character_next_LSB_plane;
                    state->background_palette_MSB_shift_register = (state->background_palette_MSB_shift_register & 0xFF00) | ((state->background_next_attribute >> 1) & 0x01)*0x00FF;
                    state->background_palette_LSB_shift_register = (state->background_palette_LSB_shift_register & 0xFF00) | ((state->background_next_attribute >> 0) & 0x01)*0x00FF;
                    break;
                case 1:
                    /* horizontal prepare */
                    if ((state->horizontal_scan == 257) && is_rendering_enabled) {
                        state->loopy_V = (state->loopy_V & ~0b0000010000011111) | (state->loopy_T & 0b0000010000011111);
                    }

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
                    state->loopy_V = ((state->loopy_V & ~0b0000000000011111) | coarse_X) ^ 0b0000010000000000*(coarse_X == 0);
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

        scanlineNext(this, state, memory);
        state->ppu_cycle++;
        scanlineNext(this, state, memory);
        state->ppu_cycle++;
        scanlineNext(this, state, memory);
        state->ppu_cycle++;

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
        if (blockIdx.x == 0 && threadIdx.x == OBSERVED_INSTANCE && instruction_OK) {
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
