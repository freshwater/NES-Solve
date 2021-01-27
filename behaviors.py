class Behaviors:
    def reset_vblank_on_read(ppu, register_index):
        """
        http://wiki.nesdev.com/w/index.php/PPU_registers#PPUSTATUS
        "Reading the status register will clear bit 7 mentioned above
        and also the address latch used by PPUSCROLL and PPUADDR.
        It does not clear the sprite 0 hit or overflow bit."
        """
        if (register_index + PPU_REGISTERS) == PPU_STATUS:
            ppu.registers.array[register_index] &= 0x7F
            ppu.address_latch = 0

    def write_special_status_bits_on_push(function, status_register):
        """
        https://wiki.nesdev.com/w/index.php/Status_flags#The_B_flag
        Two interrupts (/IRQ and /NMI) and two instructions (PHP and BRK)
        push the flags to the stack. In the byte pushed, bit 5 is always
        set to 1, and bit 4 is 1 if from an instruction (PHP or BRK) or 0
        if from an interrupt line being pulled low (/IRQ or /NMI). This is
        the only time and place where the B flag actually exists: not in
        the status register itself, but in bit 4 of the copy that is
        written to the stack.
        """
        from instructions import PHP

        return status_register | 0x20 | (0x10 if function in [PHP] else 0x00)

    def read_special_status_bits_on_pull(state, data):
        """
        https://wiki.nesdev.com/w/index.php/Status_flags#The_B_flag
        Two instructions (PLP and RTI) pull a byte from the stack and set all
        the flags. They ignore bits 5 and 4.
        """
        bits = state.status_register_byte() & 0x30
        data &= (0xFF - 0x30)
        data |= bits

        return data

