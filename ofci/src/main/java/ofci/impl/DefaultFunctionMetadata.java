package ofci.impl;

import ofci.FunctionMetadata;
import ofci.GlobalFunctionInfo;

import ghidra.program.model.listing.Function;

public class DefaultFunctionMetadata implements FunctionMetadata {
    private GlobalFunctionInfo globalInfo = new DummyGlobalFunctionInfo();
    private int nameId = -1;
    private int callCount = 0;

    public void incReferencedFuncCount() {
        callCount += 1;
    }

    public void setFunction(Function function) {
        final var id = globalInfo.getFunctionNameId(function.getEntryPoint());
        if (id != null)
            this.nameId = id;
    }

    public void clearState() {
        nameId = -1;
        callCount = 0;
    }

    public int getNameId() {
        return nameId;
    }

    public int getReferencedFuncCount() {
        return callCount;
    }

    public GlobalFunctionInfo getGlobalInfo() {
        return globalInfo;
    }

    public void setGlobalInfo(GlobalFunctionInfo globalInfo) {
        this.globalInfo = globalInfo;
    }
}
