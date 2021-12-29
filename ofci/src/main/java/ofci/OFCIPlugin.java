package ofci;

import ghidra.MiscellaneousPluginPackage;
import ghidra.app.plugin.PluginCategoryNames;
import ghidra.app.plugin.ProgramPlugin;
import ghidra.framework.options.ToolOptions;
import ghidra.framework.plugintool.PluginInfo;
import ghidra.framework.plugintool.PluginTool;
import ghidra.framework.plugintool.util.PluginStatus;
import ghidra.program.model.listing.Program;
import ghidra.program.util.ProgramLocation;
import ghidra.util.HelpLocation;
import ghidra.util.Msg;

@PluginInfo(
    status = PluginStatus.UNSTABLE,
    packageName = MiscellaneousPluginPackage.NAME,
    category = PluginCategoryNames.USER_ANNOTATION,
    shortDescription = "Obfuscated Function Clone Identification",
    description = "Uses transformer models to create function embeddings.",
    servicesProvided = { Factory.class }
)
public class OFCIPlugin extends ProgramPlugin {
    public static final String OPTIONS_TITLE = "OFCI";
    public static final String DB_URL_OPTION = "PostgreSQL JDBC URL";
    public static final String DB_URL_DEFAULT = "jdbc:postgresql://db:5432/ofci";
    public static final String DB_USER_OPTION = "PostgreSQL User";
    public static final String DB_USER_DEFAULT = "ofci";
    public static final String DB_PASS_OPTION = "PostgreSQL Password";
    public static final String DB_PASS_DEFAULT = "ofci";

    private final Factory factory;
    //private final OFCIProvider provider;

    public OFCIPlugin(PluginTool tool) {
        super(tool, true, false);

        //provider = new OFCIProvider(this);
        //provider.addToTool();

        factory = new Factory();
        registerServiceProvided(Factory.class, factory);

        initializeOptions();
    }

    @Override
    protected void programOpened(Program program) {
        //Msg.showInfo(this, tool.getActiveWindow(), "OFCI", "loaded");
        factory.setProgram(program);
    }

    @Override
    protected void locationChanged(ProgramLocation location) {
        if (location == null) {
            factory.setCurrentFunction(null);
            return;
        }
        
        final var fnMan = currentProgram.getFunctionManager();
        factory.setCurrentFunction(fnMan.getFunctionContaining(location.getAddress()));
    }

    @Override
    protected void dispose() {
        //provider.dispose();
    }

    private void initializeOptions() {
        ToolOptions opt = tool.getOptions(OPTIONS_TITLE);
        HelpLocation help = new HelpLocation(OPTIONS_TITLE, DB_URL_OPTION);
        opt.registerOption(DB_URL_OPTION, DB_URL_DEFAULT, help,
            "JDBC URL for a PostgreSQL server.");
        factory.setDbUrl(opt.getString(DB_URL_OPTION, DB_URL_DEFAULT));

        help = new HelpLocation(OPTIONS_TITLE, DB_USER_OPTION);
        opt.registerOption(DB_USER_OPTION, DB_USER_DEFAULT, help,
            "Username for a PostgreSQL database.");
        factory.setDbUser(opt.getString(DB_USER_OPTION, DB_USER_DEFAULT));

        help = new HelpLocation(OPTIONS_TITLE, DB_PASS_OPTION);
        opt.registerOption(DB_PASS_OPTION, DB_PASS_DEFAULT, help,
            "Password for a PostgreSQL database.");
        factory.setDbPass(opt.getString(DB_PASS_OPTION, DB_PASS_DEFAULT));

        opt.addOptionsChangeListener(factory);
    }
}
