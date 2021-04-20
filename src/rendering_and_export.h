
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

                    int hash_index = __float_as_int(kernel_total * kernel_total * kernel_total * kernel_total) & (HASH_SIZE-1);
                    int64_t instance_hash_index = hash_index*(gridDim.x*blockDim.x) + blockIdx.x*(blockDim.x) + threadIdx.x;
                    system->hash_sets[instance_hash_index] = 1;

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
