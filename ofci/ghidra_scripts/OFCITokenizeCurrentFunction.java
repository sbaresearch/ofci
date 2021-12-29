// Extract data for use with OFCI, uses postgres
// @author anon
// @category Analysis

import ghidra.app.script.GhidraScript;
import ghidra.util.Msg;

import ofci.Factory;

public class OFCITokenizeCurrentFunction extends GhidraScript {
    @Override
    public void run() throws Exception {
        final var factory = state.getTool().getService(Factory.class);
        factory.initializeEvaluationInfo(currentProgram.getName(), getScriptArgs());

        // Initialize parsers
        final var tokenizer = factory.createTokenizer();
        final var functionMetadata = factory.createFunctionMetadata();
        final var instructionParser = factory.createInstructionParserAllSeparators(
            tokenizer,
            functionMetadata
        );
        final var functionParser = factory.createFunctionParser(
            tokenizer,
            instructionParser,
            currentProgram.getListing()
        );

        final var function = factory.getCurrentFunction();
        if (function == null) {
            var tool = state.getTool();
            var parent = tool != null ? tool.getActiveWindow() : null;
            Msg.showError(this, parent, "Error", "No function selected.");
            return;
        }

        functionMetadata.setFunction(function);
        functionParser.addRanges(function.getBody());
        println(function.getName() + ": " + functionParser.getParsedFunction());
    }
}
