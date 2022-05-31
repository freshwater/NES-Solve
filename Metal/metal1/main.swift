//
//  main.swift
//

import MetalKit

func thing() {
    let device = MTLCreateSystemDefaultDevice()!
    let library = device.makeDefaultLibrary()

    let metalFunction = library!.makeFunction(name: "run")!

    device.makeComputePipelineState(function: metalFunction) { pipelineState, error in
        if let commandQueue = device.makeCommandQueue(),
           let commandBuffer = commandQueue.makeCommandBuffer(),
           let commandEncoder = commandBuffer.makeComputeCommandEncoder() {

            func romFileRead(fileName: String) -> (Data, Data)? {
                let data = try! Data(contentsOf: URL(fileURLWithPath: fileName))

                /*
                struct ROM {
                    char _nes[3]; 0
                    char _0x1A; 3
                    char PRG; 4
                    char CHR; 5
                    char _mapper1; 6
                    char _unused2[9]; 7
                    char etc[*]; 16
                } rom;
                */

                let PRG = Int(data[4])
                let CHR = Int(data[5])
                let headerSize = 16

                /*
                std::vector<char> program_data(file_data.begin() + sizeof(ROM),
                                               file_data.begin() + sizeof(ROM) + rom.PRG*8*2048);
                std::vector<char> character_data(file_data.begin() + sizeof(ROM) + rom.PRG*8*2048,
                                                 file_data.begin() + sizeof(ROM) + rom.PRG*8*2048 + rom.CHR*8*1024);
                */
                let programData = data[headerSize..<(headerSize+PRG*8*2048)]
                let characterData = data[(headerSize+PRG*8*2048)..<(headerSize+PRG*8*2048 + CHR*8*1024)]

                return (programData, characterData)
            }

            /*SystemState(std::vector<char>& program_data, std::vector<char>& character_data) {
                 std::copy(program_data.begin(), program_data.end(), global_memory.cartridge_memory);
                 if (program_data.size() < 0x8000) {
                     std::copy(program_data.begin(), program_data.end(), (global_memory.cartridge_memory + 0x4000));
                 }
                 std::copy(character_data.begin(), character_data.end(), global_memory.ppu_memory);
                 this->program_counter_initial = (global_memory.cartridge_memory[0xFFFD % 0x8000] << 8) | global_memory.cartridge_memory[0xFFFC % 0x8000];
                 this->stack_offset_initial = 0xFD;
             }*/
            func SystemState__initialize(buffer: MTLBuffer, programData: Data, programCounterInitial: UInt16) {
                var state = SystemState()
                state.program_counter_initial = programCounterInitial
                state.stack_offset_initial = 0xFD;

                buffer.contents().storeBytes(of: state, toByteOffset: 0, as: SystemState.self)

                let cartridgePointer = (buffer.contents() + Int(cartridge_memory_offset.rawValue)).bindMemory(to: UInt8.self, capacity: Int(cartridge_memory_size.rawValue))

                let cartridgeBufferHalf1 = UnsafeMutableBufferPointer(start: cartridgePointer, count: Int(cartridge_memory_size.rawValue / 2))
                let cartridgeBufferHalf2 = UnsafeMutableBufferPointer(start: cartridgePointer + Int(cartridge_memory_size.rawValue / 2), count: Int(cartridge_memory_size.rawValue / 2))
                _=cartridgeBufferHalf1.initialize(from: programData)
                _=cartridgeBufferHalf2.initialize(from: programData)
            }

            let systemStatesBuffer = device.makeBuffer(length: MemoryLayout<SystemState>.size, options: .storageModeShared)!
            let traceLinesBuffer = device.makeBuffer(length: MemoryLayout<Trace>.stride*0x500, options: .storageModeShared)!
            commandEncoder.setBuffer(systemStatesBuffer, offset: 0, index: 0)
            commandEncoder.setBuffer(traceLinesBuffer, offset: 0, index: 1)

            /* ----- */

            let (programData, characterData) = romFileRead(fileName: "nestest.nes")!

            SystemState__initialize(buffer: systemStatesBuffer, programData: programData, programCounterInitial: 0xC000)

            /* ----- */

            commandEncoder.setComputePipelineState(pipelineState!)

            commandEncoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                           threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

            commandEncoder.endEncoding()
            commandBuffer.commit()

            commandBuffer.waitUntilCompleted()

            let pTraceLines = traceLinesBuffer.contents().bindMemory(to: Trace.self, capacity: 1)
            tracePrint(pTraceLines: pTraceLines)
        }
    }

    Thread.sleep(forTimeInterval: 1)

    print("done.")
}

thing()
