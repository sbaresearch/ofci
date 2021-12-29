package ofci.impl;

import ofci.FunctionMetadata;
import ofci.FunctionParser;
import ofci.InstructionParser;
import ofci.Tokenizer;
import ghidra.program.model.address.AddressSetView;
import ghidra.program.model.listing.Listing;

public class DefaultFunctionParser implements FunctionParser {
    private final Tokenizer tokenizer;
    private final InstructionParser instructionParser;
    private final Listing listing;

    public DefaultFunctionParser(
        Tokenizer tokenizer,
        InstructionParser instructionParser,
        Listing listing
    ) {
        this.tokenizer = tokenizer;
        this.instructionParser = instructionParser;
        this.listing = listing;
    }

    public void addRanges(AddressSetView ranges) {
        final var instructions = listing.getInstructions(ranges, true);
        for (final var instruction : instructions)
            instructionParser.addInstruction(instruction);
    }

    public FunctionMetadata getMetadata() {
        return instructionParser.getFunctionMetadata();
    }

    public void clearState() {
        instructionParser.clearState();
        tokenizer.clearState();
    }

    public String getParsedFunction() {
        return tokenizer.getBufferContents();
    }
}
