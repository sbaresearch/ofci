package ofci;

import ghidra.program.model.listing.Function;

public interface FunctionMetadata extends ParserState {
    public void incReferencedFuncCount();

    public void setFunction(Function function);

    public int getNameId();

    public int getReferencedFuncCount();

    public GlobalFunctionInfo getGlobalInfo();

    public void setGlobalInfo(GlobalFunctionInfo globalInfo);
}
