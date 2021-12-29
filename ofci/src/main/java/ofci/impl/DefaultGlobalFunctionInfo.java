package ofci.impl;

import ofci.Dao;
import ofci.GlobalFunctionInfo;
import ofci.Factory;

import java.sql.SQLException;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;

public class DefaultGlobalFunctionInfo implements GlobalFunctionInfo {
    private final Dao dao;
    private final Map<Address, Integer> fnNameIdMap;
    private final Factory factory;

    private int categoryId = 0;
    private int groupId = 0;
    private int toolId = 0;

    public DefaultGlobalFunctionInfo(Factory factory, Dao dao) {
        this.factory = factory;
        this.dao = dao;
        this.fnNameIdMap = new HashMap<>();
    }

    public int getCategoryId() {
        return categoryId;
    }

    public int getToolId() {
        return toolId;
    }

    public int getGroupId() {
        return groupId;
    }

    public void resolveEvalNameIds() throws SQLException {
        final var categoryName = factory.getCategoryName();
        categoryId = dao.getCategoryId(categoryName);

        final var groupName = factory.getGroupName();
        groupId = dao.getGroupId(groupName);

        final var toolName = factory.getToolName();
        toolId = dao.getToolId(toolName);
    }

    public void resolveFunctionNameIds(List<Function> functions) throws SQLException {
        for (final var function : functions)
            dao.addFunctionNameBatch(function.getName());

        final var ids = dao.functionNameBatchInsert();
        for (int i = 0; i < functions.size(); i++)
            fnNameIdMap.put(functions.get(i).getEntryPoint(), ids.get(i));
    }

    public Integer getFunctionNameId(Address address) {
        return fnNameIdMap.get(address);
    }
}
