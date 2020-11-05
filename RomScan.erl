
%% https://formats.kaitai.io/ines/index.html
%% https://www.masswerk.at/6502/6502_instruction_set.html
%% https://gist.github.com/1wErt3r/4048722
%% http://archive.6502.org/books/mcs6500_family_programming_manual.pdf
%% http://bootgod.dyndns.org:7777/profile.php?id=270

-export([d/2]).

main([FileName|_] = _Args) ->

    {ok, Data} = file:read_file(FileName),

    <<"NES\x1A",
        PRG, CHR,
        _:4, _IgnoreMirroring:1, _HasTrainer:1, _HasBattery:1, _VerticalMirroring:1,
        _Mapper2,
        _Unused:64,

        PrgRom:(PRG*2048*8)/binary,
        ChrRom:(CHR*1024*8)/binary>> = Data,

    Hex = fun(I) -> integer_to_list(I, 16) end,
    _PrintLine =
        fun ({I, OP, A, B, C}) -> io:format("~4..0s ~s $~2..0s~2..0s,~s~n", [Hex(I), OP, Hex(B), Hex(A), C]);
            ({I, OP, A, "X"}) -> io:format("~4..0s ~s $~2..0s,~s~n", [Hex(I), OP, Hex(A), "X"]);
            ({I, OP, A, B}) -> io:format("~4..0s ~s $~2..0s~2..0s~n", [Hex(I), OP, Hex(B), Hex(A)]);
            ({I, OP, A}) -> io:format("~4..0s ~s $~2..0s~n", [Hex(I), OP, Hex(A)]);
            ({I, OP}) -> io:format("~4..0s ~s~n", [Hex(I), OP]) end,

    % ProgramList = binary_to_list(PrgRom),
    % PrgRomPart1 = list_to_binary(lists:sublist(ProgramList, 1, 5000)),
    % lists:foreach(PrintLine, d(37068, PrgRomPart1)),

    io:format("{\"Program\": ~w, \"Character\": ~w}",
        [binary_to_list(PrgRom), binary_to_list(ChrRom)]),

    ok.

d(I, <<"\x01", A:8,      Rest/binary>>) -> [{I, "ORA", A, "X"} | d(I+2, Rest)];
d(I, <<"\x09", A:8,      Rest/binary>>) -> [{I, "ORA", A}    | d(I+2, Rest)];

d(I, <<"\x10", A:8,      Rest/binary>>) -> [{I, "BPL", A}    | d(I+2, Rest)];
d(I, <<"\x20", A:8, B:8, Rest/binary>>) -> [{I, "JSR", A, B} | d(I+3, Rest)];
d(I, <<"\x29", A:8,      Rest/binary>>) -> [{I, "AND", A}    | d(I+2, Rest)];
d(I, <<"\x2C", A:8,      Rest/binary>>) -> [{I, "BIT", A}    | d(I+2, Rest)];

d(I, <<"\x41", A:8,      Rest/binary>>) -> [{I, "EOR", A}    | d(I+2, Rest)];
d(I, <<"\x4C", A:8, B:8, Rest/binary>>) -> [{I, "JMP", A, B} | d(I+3, Rest)];

d(I, <<"\x60",           Rest/binary>>) -> [{I, "RTS"}       | d(I+1, Rest)];

d(I, <<"\x78",           Rest/binary>>) -> [{I, "SEI"}       | d(I+1, Rest)];
d(I, <<"\x85", A:8,      Rest/binary>>) -> [{I, "STA", A}    | d(I+2, Rest)];
d(I, <<"\x86", A:8,      Rest/binary>>) -> [{I, "STX", A}    | d(I+2, Rest)];
d(I, <<"\x88",           Rest/binary>>) -> [{I, "DEY"}       | d(I+1, Rest)];
d(I, <<"\x8D", A:8, B:8, Rest/binary>>) -> [{I, "STA", A, B} | d(I+3, Rest)];

d(I, <<"\x91", A:8,      Rest/binary>>) -> [{I, "STA", A}    | d(I+2, Rest)];
d(I, <<"\x9A",           Rest/binary>>) -> [{I, "TXS"}       | d(I+1, Rest)];

d(I, <<"\xA0", A:8,      Rest/binary>>) -> [{I, "LDY", A}    | d(I+2, Rest)];
d(I, <<"\xA2", A:8,      Rest/binary>>) -> [{I, "LDX", A}    | d(I+2, Rest)];
d(I, <<"\xA9", A:8,      Rest/binary>>) -> [{I, "LDA", A}    | d(I+2, Rest)];
d(I, <<"\xAC", A:8, B:8, Rest/binary>>) -> [{I, "LDY", A, B} | d(I+3, Rest)];
d(I, <<"\xAD", A:8, B:8, Rest/binary>>) -> [{I, "LDA", A, B} | d(I+3, Rest)];

d(I, <<"\xB0", A:8,      Rest/binary>>) -> [{I, "BCS", A}    | d(I+2, Rest)];
d(I, <<"\xBD", A:8, B:8, Rest/binary>>) -> [{I, "LDA", A, B, "X"} | d(I+3, Rest)];

d(I, <<"\xC0", A:8,      Rest/binary>>) -> [{I, "CPY", A}       | d(I+2, Rest)];
d(I, <<"\xC8",           Rest/binary>>) -> [{I, "INY"}       | d(I+1, Rest)];
d(I, <<"\xC9", A:8,      Rest/binary>>) -> [{I, "CMP", A}    | d(I+2, Rest)];
d(I, <<"\xCA",           Rest/binary>>) -> [{I, "DEX"}       | d(I+1, Rest)];

d(I, <<"\xD0", A:8,      Rest/binary>>) -> [{I, "BNE", A}    | d(I+2, Rest)];
d(I, <<"\xD8",           Rest/binary>>) -> [{I, "CLD"}       | d(I+1, Rest)];

d(I, <<"\xE0", A:8,      Rest/binary>>) -> [{I, "CPX", A}    | d(I+2, Rest)];
d(I, <<"\xEE", A:8, B:8, Rest/binary>>) -> [{I, "INC", A, B} | d(I+3, Rest)];

d(I, <<OP:8, A:8, B:8, _Rest/binary>>) ->
    io:format("~w [~s|~2..0s.~2..0s]~n~n", [I, integer_to_list(OP, 16),
                                            integer_to_list(A, 16),
                                            integer_to_list(B, 16)]),

    [];

d(_I, _Rest) ->
    %% io:format("[~p~n~p]~n~n", [I, _Rest]),
    [].