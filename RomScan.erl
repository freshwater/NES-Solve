
%% https://formats.kaitai.io/ines/index.html
%% https://www.masswerk.at/6502/6502_instruction_set.html
%% http://archive.6502.org/books/mcs6500_family_programming_manual.pdf

main([FileName|_] = _Args) ->

    {ok, Data} = file:read_file(FileName),

    <<"NES\x1A",
        PRG, CHR,
        _:4, IgnoreMirroring:1, HasTrainer:1, HasBattery:1, VerticalMirroring:1,
        Mapper2,
        Unused:64,
        Rest/binary>> = Data,

    Prg = PRG*2048*8,
    Chr = CHR*1024*8,

    <<PrgRom:Prg/binary,
      ChrRom:Chr/binary>> = Rest,

    io:format("{\"Program\": ~w, \"Character\": ~w}",
        [binary_to_list(PrgRom), binary_to_list(ChrRom)]),

    ok.

