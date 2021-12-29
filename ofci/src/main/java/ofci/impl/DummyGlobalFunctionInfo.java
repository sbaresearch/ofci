package ofci.impl;

import ofci.GlobalFunctionInfo;

import java.util.List;

import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;

public class DummyGlobalFunctionInfo implements GlobalFunctionInfo {
    public int getCategoryId() { return 0; }

    public int getToolId() { return 0; }

    public int getGroupId() { return 0; }

    public void resolveEvalNameIds() { }

    public void resolveFunctionNameIds(List<Function> functions) { }

    public Integer getFunctionNameId(Address address) { return null; }
}
