package ofci.impl;

import ofci.Tokenizer;
import ofci.FunctionMetadata;

import ghidra.program.model.listing.Instruction;

public class InstructionParserAllSeparators extends DefaultInstructionParser {
    public InstructionParserAllSeparators(
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

    protected void addInstructionSeparator() {
        tokenizer.addChar('.');
    }
}
