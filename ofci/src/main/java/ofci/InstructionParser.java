package ofci;

import ghidra.program.model.listing.Instruction;

public interface InstructionParser extends ParserState {
    public void addInstruction(Instruction instruction);

    public FunctionMetadata getFunctionMetadata();
}
