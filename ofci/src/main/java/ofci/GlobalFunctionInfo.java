package ofci;

import java.sql.SQLException;
import java.util.List;

import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;

public interface GlobalFunctionInfo {
    public int getCategoryId();

    public int getToolId();

    public int getGroupId();

    public void resolveEvalNameIds() throws SQLException;

    public void resolveFunctionNameIds(List<Function> functions) throws SQLException;

    public Integer getFunctionNameId(Address address);
}
