package ofci;

import java.awt.BorderLayout;
import javax.swing.JComponent;
import javax.swing.JPanel;

import docking.ActionContext;
import docking.action.DockingAction;
import docking.action.ToolBarData;
import ghidra.framework.plugintool.ComponentProviderAdapter;
import ghidra.util.Msg;
//import resources.Icons;

public class OFCIProvider extends ComponentProviderAdapter {
    private OFCIPlugin plugin;
    private JComponent component;
    private DockingAction tokenizeCurrentFunction;

    public OFCIProvider(OFCIPlugin plugin) {
        super(plugin.getTool(), "OFCI Provider", plugin.getName());
        this.plugin = plugin;

        component = build();
        createActions();
    }

    void dispose() {
        removeFromTool();
    }

    private JComponent build() {
        JPanel panel = new JPanel(new BorderLayout());
        return panel;
    }

    private void createActions() {
        var name = "Tokenize Current Function";
        tokenizeCurrentFunction = new DockingAction(name, plugin.getName()) {
            @Override
            public void actionPerformed(ActionContext context) {
                Msg.showInfo(getClass(), component, "Test", "hello");
            }
        };
        //tokenizeCurrentFunction.setToolBarData(new ToolBarData(Icons.ADD_ICON, null));
        tokenizeCurrentFunction.setEnabled(true);
        tokenizeCurrentFunction.markHelpUnnecessary();
        dockingTool.addLocalAction(this, tokenizeCurrentFunction);
        plugin.getTool().addAction(tokenizeCurrentFunction);
    }

    @Override
    public JComponent getComponent() {
        return component;
    }
}
