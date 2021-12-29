package ofci.impl;

import ofci.Dao;
import ofci.FunctionMetadata;
import ofci.FunctionParser;
import ofci.GlobalFunctionInfo;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class DefaultDao implements Dao {
    private final Connection conn;
    private final PreparedStatement[] stmts;
    private int numNameBatches;

    private final static int STMT_GROUP = 0;
    private final static int STMT_TOOL = 1;
    private final static int STMT_CATEGORY = 2;
    private final static int STMT_FUNCNAME = 3;
    private final static int STMT_FUNC = 4;

    private final static String[] stmtStrings = {
        "INSERT INTO group_names (name) VALUES (?) " +
        "ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name " +
        "RETURNING id",
        "INSERT INTO tool_names (name) VALUES (?) " +
        "ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name " +
        "RETURNING id",
        "INSERT INTO category_names (name) VALUES (?) " +
        "ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name " +
        "RETURNING id",
        "INSERT INTO function_names (name) VALUES (?) " +
        "ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name " +
        "RETURNING id",
        "INSERT INTO functions (name,project_group,tool," +
        "category,call_count,hash,contents) VALUES " +
        "(?,?,?,?,?,?,?) RETURNING id"
    };
    
    private final static String[] returnIdColumns = { "id" };

    public DefaultDao(Connection conn) {
        this.conn = conn;
        this.stmts = new PreparedStatement[stmtStrings.length];
        this.numNameBatches = 0;
    }

    public void prepareStatements() throws SQLException {
        for (int i = 0; i < stmtStrings.length; i++)
            stmts[i] = conn.prepareStatement(stmtStrings[i], returnIdColumns);
    }

    public int getGroupId(String groupName) throws SQLException {
        final var stmt = stmts[STMT_GROUP];
        stmt.setString(1, groupName);
        stmt.executeUpdate();
        try (final var rs = stmt.getGeneratedKeys()) {
            rs.next();
            return rs.getInt(1);
        }
    }

    public int getToolId(String toolName) throws SQLException {
        final var stmt = stmts[STMT_TOOL];
        stmt.setString(1, toolName);
        stmt.executeUpdate();
        try (final var rs = stmt.getGeneratedKeys()) {
            rs.next();
            return rs.getInt(1);
        }
    }

    public int getCategoryId(String categoryName) throws SQLException {
        final var stmt = stmts[STMT_CATEGORY];
        stmt.setString(1, categoryName);
        stmt.executeUpdate();
        try (final var rs = stmt.getGeneratedKeys()) {
            rs.next();
            return rs.getInt(1);
        }
    }

    public int getFunctionId(String functionName) throws SQLException {
        final var stmt = stmts[STMT_FUNCNAME];
        stmt.setString(1, functionName);
        stmt.executeUpdate();
        try (final var rs = stmt.getGeneratedKeys()) {
            rs.next();
            return rs.getInt(1);
        }
    }

    public void addFunctionNameBatch(String functionName) throws SQLException {
        final var stmt = stmts[STMT_FUNCNAME];
        stmt.clearParameters();
        stmt.setString(1, functionName);
        stmt.addBatch();
        numNameBatches += 1;
    }

    public void addFunctionBatch(FunctionParser functionParser) throws SQLException {
        final var stmt = stmts[STMT_FUNC];
        final var fnContent = functionParser.getParsedFunction();
        final var functionMetadata = functionParser.getMetadata();
        final var globalFunctionInfo = functionMetadata.getGlobalInfo();
        stmt.clearParameters();
        stmt.setInt(1, functionMetadata.getNameId());
        stmt.setInt(2, globalFunctionInfo.getGroupId());
        stmt.setInt(3, globalFunctionInfo.getToolId());
        stmt.setInt(4, globalFunctionInfo.getCategoryId());
        stmt.setInt(5, functionMetadata.getReferencedFuncCount());
        stmt.setObject(6, UUID.nameUUIDFromBytes(fnContent.getBytes()));
        stmt.setString(7, fnContent);
        stmt.addBatch();
    }

    public List<Integer> functionNameBatchInsert() throws SQLException {
        final var stmt = stmts[STMT_FUNCNAME];
        final var nameIds = new ArrayList<Integer>(numNameBatches);
        stmt.clearParameters();
        stmt.executeBatch();
        numNameBatches = 0;

        try (final var rs = stmt.getGeneratedKeys()) {
            while (rs.next())
                nameIds.add(rs.getInt(1));
        }

        return nameIds;
    }

    public void functionBatchInsert() throws SQLException {
        final var stmt = stmts[STMT_FUNC];
        stmt.clearParameters();
        stmt.executeBatch();
    }

    public void close() throws Exception {
        for (final var stmt : stmts)
            stmt.close();
        conn.close();
    }
}
