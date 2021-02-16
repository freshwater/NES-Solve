
std::string traceLineFormat(Trace trace)
{
    uint8_t opcode = unsigned((uint8_t)trace.opcode);
    OperationInformation info = operation_info[opcode];

    char str[4][100];

    sprintf(str[0], "%04X %02X %02X",
            trace.ProgramCounter,
            unsigned((uint8_t)trace.opcode),
            unsigned((uint8_t)trace.byte1));

    if (info.byte_count == 2) {
        strcpy(str[1], "");
    } else if (info.byte_count == 3) {
        sprintf(str[1], " %02X",
            unsigned((uint8_t)trace.byte2));
    } else {
        strcpy(str[1], " --");
    }

    sprintf(str[2], " %s %s", info.name.data(),
            info.doFormat(trace.byte1, trace.byte2).data());

    sprintf(str[3], " A:%02X X:%02X Y:%02X",
            unsigned((uint8_t)trace.A),
            unsigned((uint8_t)trace.X),
            unsigned((uint8_t)trace.Y));

    using namespace std;
    return string(str[0]) + string(str[1]) + string(str[2]) + string(str[3]);
}

std::string logLineFormat(std::vector<std::string> line)
{
    std::string hex("0123456789ABCDEF");
    int opcode = hex.find(line[1][0])*16 + hex.find(line[1][1]);

    int ppuI = -1;
    while (line[++ppuI] != "PPU:");

    line = std::vector<std::string>(line.begin(), line.begin()+ppuI);
    std::vector<std::string> reduced;

    OperationInformation info = operation_info[opcode];
    int j = info.format_type == "Implied" ? 2 : 3;
    for (int i = 0; i < line.size(); i++) {
        if ((i < j + info.byte_count) | (line.size() - 6 < i)) {
            reduced.push_back(line[i]);
        }
    }

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

std::vector<std::vector<std::string>> logRead(std::string file_name)
{
    std::ifstream file(file_name);
    std::vector<std::vector<std::string>> output;

    std::string line_string;

    while (std::getline(file, line_string)) {
        std::vector<std::string> words;

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
