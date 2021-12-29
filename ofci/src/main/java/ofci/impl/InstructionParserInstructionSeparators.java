package ofci.impl;

import ofci.Tokenizer;
import ofci.FunctionMetadata;

import ghidra.program.model.listing.Instruction;

public class InstructionParserInstructionSeparators extends DefaultInstructionParser {
    public InstructionParserInstructionSeparators(
        Tokenizer tokenizer,
        FunctionMetadata functionMetadata
    ) {
        super(tokenizer, functionMetadata);
    }

    protected void addOperandSeparator(Instruction instruction, int index) { }

    protected void addInstructionSeparator() {
        tokenizer.addChar('.');
    }
}
