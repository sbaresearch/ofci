// Extract data for use with OFCI, uses postgres
// @author anon
// @category Analysis

import java.util.ArrayList;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.util.Msg;

import ofci.Factory;

public class OFCIExtractorV2 extends GhidraScript {
    @Override
    public void run() throws Exception {
        // Initialize factory and check if evaluation values are present
        final var factory = isRunningHeadless() ?
            new Factory(currentProgram) : state.getTool().getService(Factory.class);
        factory.initializeEvaluationInfo(currentProgram.getName(), getScriptArgs());
        if (!factory.isEvaluationInfoSet()) {
            if (isRunningHeadless()) {
                var msg = "Evaluation info (group, category, tool) not set. ";
                msg += "Is the filename in the correct format and category in args?";
                Msg.error(this, msg);
                return;
            } else {
                var tool = state.getTool();
                var parent = tool != null ? tool.getActiveWindow() : null;
                var msg = "Evaluation info (group, category, tool) not set. ";
                msg += "Consider adding it under Edit -> \"Options for '";
                msg += currentProgram.getName();
                msg += "'\".";
                Msg.showError(this, parent, "Error", msg);
                return;
            }
        }

        // Load functions
        final var fnMan = currentProgram.getFunctionManager();
        final var functions = new ArrayList<Function>();
        for (final var function : fnMan.getFunctionsNoStubs(true))
            functions.add(function);

        // Initialize parsers
        final var tokenizer = factory.createTokenizer();
        final var functionMetadata = factory.createFunctionMetadata();
        final var instructionParser = factory.createInstructionParserInstructionSeparators(
            tokenizer,
            functionMetadata
        );
        final var functionParser = factory.createFunctionParser(
            tokenizer,
            instructionParser,
            currentProgram.getListing()
        );

        try(final var dao = factory.createDao()) {
            final var globalFunctionInfo = factory.createGlobalFunctionInfo(dao);

            // Initialize function info
            globalFunctionInfo.resolveEvalNameIds();
            globalFunctionInfo.resolveFunctionNameIds(functions);
            functionMetadata.setGlobalInfo(globalFunctionInfo);

            monitor.initialize(functions.size());
            for (final var function : functions) {
                if (function.isExternal() || function.isThunk())
                    continue;

                functionMetadata.setFunction(function);
                functionParser.addRanges(function.getBody());
                dao.addFunctionBatch(functionParser);
                functionParser.clearState();
                monitor.incrementProgress(1);
            }

            dao.functionBatchInsert();
        }
    }
}
