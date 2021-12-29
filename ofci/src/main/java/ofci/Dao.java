package ofci;

import java.sql.SQLException;
import java.util.List;

public interface Dao extends AutoCloseable {
    public int getGroupId(String groupName) throws SQLException;

    public int getToolId(String toolName) throws SQLException;

    public int getCategoryId(String categoryName) throws SQLException;

    public int getFunctionId(String functionName) throws SQLException;

    public void addFunctionNameBatch(String functionName) throws SQLException;

    public void addFunctionBatch(FunctionParser functionParser) throws SQLException;

    public List<Integer> functionNameBatchInsert() throws SQLException;

    public void functionBatchInsert() throws SQLException;
}
