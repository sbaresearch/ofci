package ofci;

import ofci.impl.DefaultTokenizer;
import ofci.impl.InstructionParserAllSeparators;
import ofci.impl.InstructionParserOperandSeparators;
import ofci.impl.InstructionParserInstructionSeparators;
import ofci.impl.DefaultFunctionMetadata;
import ofci.impl.DefaultFunctionParser;
import ofci.impl.DefaultDao;
import ofci.impl.DefaultGlobalFunctionInfo;

import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.regex.Pattern;

import org.postgresql.Driver;

import ghidra.app.script.GhidraState;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.listing.Program;
import ghidra.framework.options.OptionsChangeListener;
import ghidra.framework.options.ToolOptions;
import ghidra.util.HelpLocation;

public class Factory implements OptionsChangeListener {
    private static final String FILE_REGEX = "([^-]+)-([0-9][^-]+(-[0-9][^-]+)?)-([^-]+(-[^-]+)?(-[^-]+)?(-[^-]+)?)";
    private static final Pattern FILE_PATTERN = Pattern.compile(FILE_REGEX);

    private static final String PROG_TOOL_OPTION = "Tool Name";
    private static final String PROG_TOOL_DEFAULT = "";
    private static final String PROG_GROUP_OPTION = "Group Name";
    private static final String PROG_GROUP_DEFAULT = "";
    private static final String PROG_CATEGORY_OPTION = "Category Name";
    private static final String PROG_CATEGORY_DEFAULT = "";

    private Program program;
    private Function currentFunction;
    private String dbUrl = OFCIPlugin.DB_URL_DEFAULT;
    private String dbUser = OFCIPlugin.DB_USER_DEFAULT;
    private String dbPass = OFCIPlugin.DB_PASS_DEFAULT;

    public Factory() { }

    public Factory(Program program) {
        this.program = program;
        registerProgramOptions();
    }

    public void setProgram(Program program) {
        this.program = program;
        registerProgramOptions();
    }

    public void initializeEvaluationInfo(String programName, String[] scriptArgs) {
        // Don't do anything if program already has info saved
        if (isEvaluationInfoSet())
            return;

        // Don't do anything if no category is provided in args
        if (scriptArgs.length < 1)
            return;

        // Don't do anyting if regex doesn't match filename
        var matcher =  FILE_PATTERN.matcher(programName);
        if (!matcher.matches())
            return;
        
        // Calculate tool string (because of the dataset naming..)
        String[] archiveStr = matcher.group(4).split("_ARCHIVE_", 2);
        String[] dotStr = archiveStr[0].split("\\.", 2);

        final var toolString = dotStr[0];
        final var groupString = matcher.group(1);
        final var categoryName = scriptArgs[0];

        setToolName(toolString);
        setGroupName(groupString);
        setCategoryName(categoryName);
    }

    public boolean isEvaluationInfoSet() {
        return getToolName() != null &&
            !getToolName().isEmpty() &&
            getGroupName() != null &&
            !getGroupName().isEmpty() &&
            getCategoryName() != null &&
            !getCategoryName().isEmpty();
    }

    public Tokenizer createTokenizer() {
        return new DefaultTokenizer();
    }

    public FunctionMetadata createFunctionMetadata() {
        return new DefaultFunctionMetadata();
    }

    public InstructionParser createInstructionParserAllSeparators(
        Tokenizer tokenizer,
        FunctionMetadata functionMetadata
    ) {
        return new InstructionParserAllSeparators(
            tokenizer,
            functionMetadata
        );
    }

    public InstructionParser createInstructionParserOperandSeparators (
        Tokenizer tokenizer,
        FunctionMetadata functionMetadata
    ) {
        return new InstructionParserOperandSeparators(
            tokenizer,
            functionMetadata
        );
    }

    public InstructionParser createInstructionParserInstructionSeparators (
        Tokenizer tokenizer,
        FunctionMetadata functionMetadata
    ) {
        return new InstructionParserInstructionSeparators(
            tokenizer,
            functionMetadata
        );
    }

    public FunctionParser createFunctionParser(
        Tokenizer tokenizer,
        InstructionParser instructionParser,
        Listing listing
    ) {
        return new DefaultFunctionParser(
            tokenizer,
            instructionParser,
            listing
        );
    }

    public GlobalFunctionInfo createGlobalFunctionInfo(Dao dao) {
        return new DefaultGlobalFunctionInfo(this, dao);
    }

    public Dao createDao() throws SQLException {
        final var conn = DriverManager.getConnection(dbUrl, dbUser, dbPass);
        final var dao = new DefaultDao(conn);
        dao.prepareStatements();
        return dao;
    }

    public void setDbUrl(String dbUrl) {
        this.dbUrl = dbUrl;
    }

    public void setDbUser(String dbUser) {
        this.dbUser = dbUser;
    }

    public void setDbPass(String dbPass) {
        this.dbPass = dbPass;
    }

    public void setCategoryName(String categoryName) {
        final var opt = program.getOptions(OFCIPlugin.OPTIONS_TITLE);
        final var t = program.startTransaction("Updated OFCI category name");
        opt.setString(PROG_CATEGORY_OPTION, categoryName);
        program.endTransaction(t, true);
    }

    public void setGroupName(String groupName) {
        final var opt = program.getOptions(OFCIPlugin.OPTIONS_TITLE);
        final var t = program.startTransaction("Updated OFCI group name");
        opt.setString(PROG_GROUP_OPTION, groupName);
        program.endTransaction(t, true);
    }

    public void setToolName(String toolName) {
        final var opt = program.getOptions(OFCIPlugin.OPTIONS_TITLE);
        final var t = program.startTransaction("Updated OFCI tool name");
        opt.setString(PROG_TOOL_OPTION, toolName);
        program.endTransaction(t, true);
    }

    public String getToolName() {
        final var opt = program.getOptions(OFCIPlugin.OPTIONS_TITLE);
        return opt.getString(PROG_TOOL_OPTION, PROG_TOOL_DEFAULT);
    }

    public String getGroupName() {
        final var opt = program.getOptions(OFCIPlugin.OPTIONS_TITLE);
        return opt.getString(PROG_GROUP_OPTION, PROG_GROUP_DEFAULT);
    }

    public String getCategoryName() {
        final var opt = program.getOptions(OFCIPlugin.OPTIONS_TITLE);
        return opt.getString(PROG_CATEGORY_OPTION, PROG_CATEGORY_DEFAULT);
    }

    public void setCurrentFunction(Function function) {
        currentFunction = function;
    }

    public Function getCurrentFunction() {
        return currentFunction;
    }

    private void registerProgramOptions() {
        // Check if we have our per-program info set
        final var opt = program.getOptions(OFCIPlugin.OPTIONS_TITLE);
        opt.registerOption(
            PROG_CATEGORY_OPTION,
            PROG_CATEGORY_DEFAULT,
            new HelpLocation(OFCIPlugin.OPTIONS_TITLE, PROG_CATEGORY_OPTION),
            "Name of the current category within the OFCI evaluation framework."
        );
        opt.registerOption(
            PROG_GROUP_OPTION,
            PROG_GROUP_DEFAULT,
            new HelpLocation(OFCIPlugin.OPTIONS_TITLE, PROG_GROUP_OPTION),
            "Name of the current group within the OFCI evaluation framework."
        );
        opt.registerOption(
            PROG_TOOL_OPTION,
            PROG_TOOL_DEFAULT,
            new HelpLocation(OFCIPlugin.OPTIONS_TITLE, PROG_TOOL_OPTION),
            "Name of the current tool within the OFCI evaluation framework."
        );
    }

    // Options changed callback
    @Override
    public void optionsChanged(
        ToolOptions options,
        String optionName,
        Object oldValue,
        Object newValue
    ) {
        switch (optionName) {
        case OFCIPlugin.DB_URL_OPTION:
            dbUrl = (String)newValue;
            break;
        case OFCIPlugin.DB_USER_OPTION:
            dbUser = (String)newValue;
            break;
        case OFCIPlugin.DB_PASS_OPTION:
            dbPass = (String)newValue;
            break;
        default:
        }
    }
}
