package ofci;

import ghidra.program.model.address.AddressSetView;

public interface FunctionParser extends ParserState {
    public void addRanges(AddressSetView ranges);

    public FunctionMetadata getMetadata();

    public String getParsedFunction();
}
