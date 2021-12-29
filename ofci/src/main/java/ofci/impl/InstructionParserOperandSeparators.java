package ofci.impl;

import ofci.Tokenizer;
import ofci.FunctionMetadata;

import ghidra.program.model.listing.Instruction;

public class InstructionParserOperandSeparators extends DefaultInstructionParser {
    public InstructionParserOperandSeparators(
        Tokenizer tokenizer,
        FunctionMetadata functionMetadata
    ) {
        super(tokenizer, functionMetadata);
    }

    protected void addOperandSeparator(Instruction instruction, int index) {
        final var sep = instruction.getSeparator(index + 1);
        if (sep != null)
            tokenizer.addString(sep.trim());
    }

    protected void addInstructionSeparator() { }
}
