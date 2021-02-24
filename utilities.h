
std::string traceLineFormat(Trace trace, bool aligned = false)
{
    uint8_t opcode = unsigned((uint8_t)trace.opcode);
    OperationInformation info = operation_info[opcode];

    char str[6][100];
    for (char* s: str) {
        strcpy(s, "");
    }

    sprintf(str[0], "%04X %02X",
            trace.program_counter,
            unsigned((uint8_t)trace.opcode));

    if (info.byte_count > 1) {
        sprintf(str[1], " %02X",
            unsigned((uint8_t)trace.byte1));
    } else if (aligned) {
        strcpy(str[1], "   ");
    }

    if (info.byte_count > 2) {
        sprintf(str[2], " %02X",
            unsigned((uint8_t)trace.byte2));
    } else if (aligned) {
        strcpy(str[2], "   ");
    }

    if (info.byte_count < 1 | info.byte_count > 3) {
        strcpy(str[2], " --");
    }

    sprintf(str[3], " %s %s%s", info.name.data(),
            info.doFormat(trace.byte1, trace.byte2, trace.program_counter).data(), aligned ? "\t" : "");

    sprintf(str[4], "A:%02X X:%02X Y:%02X",
            unsigned((uint8_t)trace.A),
            unsigned((uint8_t)trace.X),
            unsigned((uint8_t)trace.Y));

    sprintf(str[5], " P:%02X SP:%02X PPU:[%d,%d] CYC:%d",
            unsigned((uint8_t)trace.status_register),
            unsigned((uint8_t)trace.stack_offset),
            unsigned((int16_t)trace.vertical_scan),
            unsigned((int16_t)trace.horizontal_scan),
            unsigned((uint16_t)trace.cycle));

    using namespace std;
    return string(str[0]) + string(str[1]) + string(str[2]) + string(str[3]) + string(str[4]) + string(str[5]);
}

std::string logLineFormat(std::vector<std::string> line)
{
    std::string hex("0123456789ABCDEF");
    int opcode = hex.find(line[1][0])*16 + hex.find(line[1][1]);

    int ppuI = -1;
    while (line[++ppuI] != "PPU:");

    std::vector<std::string> line1 = std::vector<std::string>(line.begin(), line.begin()+ppuI);
    std::vector<std::string> reduced;

    std::vector<std::string> line2 = std::vector<std::string>(line.begin()+ppuI, line.end());

    OperationInformation info = operation_info[opcode];
    int j = info.format_type == "Implied" ? 2 : 3;
    for (int i = 0; i < line1.size(); i++) {
        if ((i < j + info.byte_count) | (line1.size() - 6 < i)) {
            reduced.push_back(line1[i]);
        }
    }

    reduced.push_back(line2[0] + "[" + line2[1] + "," + line2[2] + "] " + line2[3]);

    std::string output;
    for (int i = 0; i < reduced.size(); i++) {
        output += " " + reduced[i];
    }
    output.erase(output.begin());

    return output;
}

std::string lineCompare(std::string line1, std::string line2)
{
    while (line1.size() < line2.size()) {
        line1 += " ";
    }
    
    while (line2.size() < line1.size()) {
        line2 += ".";
    }

    std::string output;
    for (int i = 0; i < line1.size(); i++) {
        if (line1[i] == line2[i]) {
            output += line2[i];
        } else {
            output += std::string("\033[0;31m") + line2[i] + std::string("\033[0m");
        }
    }

    return output;
}

std::vector<char> fileRead(std::string file_name)
{
    std::ifstream file(file_name, std::ios::binary);

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> file_data(fileSize);
    file.read((char*) &file_data[0], fileSize);
    file.close();

    return file_data;
}

std::pair<std::vector<char>, std::vector<char>> romFileRead(std::string file_name)
{
    std::ifstream file(file_name, std::ios::binary);

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> file_data(fileSize);
    file.read((char*) &file_data[0], fileSize);
    file.close();

    struct ROM {
        char _nes[3];
        char _0x1A;
        char PRG;
        char CHR;
        char _mapper1;
        char _unused2[9];
    } rom;

    memcpy((char*)&rom, file_data.data(), sizeof(ROM));

    std::vector<char> program_data(file_data.begin() + sizeof(ROM),
                                   file_data.begin() + sizeof(ROM) + rom.PRG*8*2048);

    std::vector<char> character_data(file_data.begin() + sizeof(ROM) + rom.PRG*8*2048,
                                     file_data.begin() + sizeof(ROM) + rom.PRG*8*2048 + rom.CHR*8*1024);

    return {program_data, character_data};
}

std::vector<std::vector<std::string>> logRead(std::string file_name)
{
    std::ifstream file(file_name);
    std::vector<std::vector<std::string>> output;

    std::string line_string;

    while (std::getline(file, line_string)) {
        std::vector<std::string> words;

        for (int i = 40; i < line_string.size(); i++) {
            if (line_string[i] == ',') {
                line_string[i] = ' ';
            }
        }

        std::istringstream ss(line_string);
        std::string word;
        while(ss >> word) {
            words.push_back(word);
        }

        output.push_back(words);
    }

    file.close();

    return output;
}
