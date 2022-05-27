//
//  Utilities.swift
//

import Foundation

func hex(_ s: UInt8) -> String { return String(format: "%02X", s) }
func hex(_ s: UInt16) -> String { return String(format: "%04X", s) }

func argumentFormat(formatType: String, programCounter: UInt16, data1: UInt8, data2: UInt8) -> String? {
    switch formatType {
        case "Absolute":
            return "$" + hex(data2) + hex(data1)
        case "Immediate":
            return "#$" + hex(data1)
        case "Zeropage":
            return "$" + hex(data1)
        case "Implied":
            return nil
        case "Address_Relative":
            return "$" + hex(programCounter + 2 + UInt16(data1))
        case "ZeropageDereference":
            return "$" + hex(data1)
        default:
            return "-" + formatType
    }
}

func tracePrint(pTraceLines: UnsafeMutablePointer<Trace>) {
    let log = try! String(contentsOf: URL(fileURLWithPath: "/Users/amr/Desktop/Projects/clang2/data/nestest.log"))
    let logLines = log.components(separatedBy: .newlines).filter {$0 != ""}

    for (i, logLine) in logLines[..<57].enumerated() {
        let line = (pTraceLines + i).pointee
        var logLineStrings = logLine.components(separatedBy: .whitespaces).filter {$0 != ""}

        if let eq = logLineStrings.firstIndex(of: "=") {
            logLineStrings.removeSubrange(eq...(eq+1))
        }

        if let ppuIndex = logLineStrings.firstIndex(of: "PPU:") {
            if logLineStrings.count - ppuIndex == 4 {
                logLineStrings = logLineStrings[...(ppuIndex-1)] +
                                   [logLineStrings[ppuIndex] + logLineStrings[ppuIndex+1] + logLineStrings[ppuIndex+2]] +
                                   logLineStrings[(ppuIndex+3)...]
            }
            if logLineStrings.count - ppuIndex == 3 {
                logLineStrings = logLineStrings[...(ppuIndex-1)] +
                                   [logLineStrings[ppuIndex] + logLineStrings[ppuIndex+1]] +
                                   logLineStrings[(ppuIndex+2)...]
            }
        }

        logLineStrings = Array(logLineStrings[0..<(logLineStrings.count - 2)])

        let (name, byteCount, formatType) = instructionsInformation[Int(line.opcode)]!

        var lineStrings = [hex(line.program_counter), hex(line.opcode)]
        lineStrings += byteCount > 1 ? [hex(line.byte1)] : []
        lineStrings += byteCount > 2 ? [hex(line.byte2)] : []

        lineStrings.append(name)
        if let arg = argumentFormat(formatType: formatType, programCounter: line.program_counter,
                                    data1: line.byte1, data2: line.byte2) {
            lineStrings.append(arg)
        }

        lineStrings.append("A:" + hex(line.A))
        lineStrings.append("X:" + hex(line.X))
        lineStrings.append("Y:" + hex(line.Y))
        lineStrings.append("P:" + hex(line.status_register))
        lineStrings.append("SP:" + hex(line.stack_offset))
        // lineStrings.append("PPU:" + "0," + String(line.horizontal_scan))

        let len = zip(lineStrings, logLineStrings).prefix(while: (==)).count
        let logBlanked = logLineStrings.enumerated().map { $0 < len ? $1.map({_ in " "}).joined() : $1 }
        print(hex(UInt16(i)) + "  " + lineStrings.joined(separator: "  ") + "  ", terminator: "")
        print((lineStrings == logLineStrings ? "" : "\n") + "    " + "  " + logBlanked.joined(separator: "  ") + "  ")

        for (l1, l2) in zip(lineStrings, logLineStrings) {
            if l1 == l2 {
                // print("(" + l1 + ")", terminator: " ")
            } else {
                print("\n<" + l1 + ", " + l2 + ">", terminator: " ")
            }
        }
        if lineStrings != logLineStrings {
            print()
            print()
        }
    }

}