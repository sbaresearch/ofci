// Extract data for use with OFCI, uses postgres
// @author anon
// @category Analysis

import java.io.File;
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.AddressSet;
import ghidra.program.model.address.AddressSetView;
import ghidra.program.model.listing.Function;
import ghidra.util.Msg;

import ofci.Factory;
import ofci.FunctionMetadata;
import ofci.FunctionParser;

public class OFCITraceExtractor extends GhidraScript {
    private final Map<Function, FunctionData> fnDataMap = new HashMap<>();
   
    private class FunctionData {
        private final FunctionMetadata functionMetadata;
        private final FunctionParser functionParser;

        private FunctionData(FunctionMetadata functionMetadata, FunctionParser functionParser) {
            this.functionMetadata = functionMetadata;
            this.functionParser = functionParser;
        }
    }

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

        try(final var dao = factory.createDao()) {
            final var globalFunctionInfo = factory.createGlobalFunctionInfo(dao);

            // Initialize function info
            globalFunctionInfo.resolveEvalNameIds();
            globalFunctionInfo.resolveFunctionNameIds(functions);

            // Get trace data
            var traceBBL = getTraceInfo(currentProgram.getName());
            
            // Add all basic blocks to functions
            monitor.initialize(traceBBL.size());
            for (var bbl : traceBBL) {
                var fn = fnMan.getFunctionContaining(bbl.getMinAddress());
                if (fn == null) {
                    var msg = "ERROR: Could not find function for BBL: ";
                    msg += bbl.getMinAddress();
                    Msg.error(this, msg);
                    continue;
                }

                if (fn.isExternal() || fn.isThunk())
                    continue;

                var fnData = fnDataMap.get(fn);
                if (fnData == null) {
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

                    functionMetadata.setGlobalInfo(globalFunctionInfo);
                    functionMetadata.setFunction(fn);

                    fnData = new FunctionData(functionMetadata, functionParser);
                    fnDataMap.put(fn, fnData);
                }

                // Add bbl to specific function
                fnData.functionParser.addRanges(bbl);
                monitor.incrementProgress(1);
            }

            // Insert all functions into DB
            monitor.initialize(fnDataMap.size());
            for (final var entry : fnDataMap.entrySet()) {
                dao.addFunctionBatch(entry.getValue().functionParser);
                monitor.incrementProgress(1);
            }
            dao.functionBatchInsert();
        }
    }

    List<AddressSetView> getTraceInfo(String name) throws Exception {
        var traceFile = isRunningHeadless() ?
            new File("/tmp/traces/" + name) : 
            askFile("Trace File Selection", "Select Trace File");
        var baseAddress = currentProgram.getImageBase();
        var traceBytes = Files.readAllBytes(Paths.get(traceFile.getPath()));
        var traceBytesLen = traceBytes.length;
        var traceBb = ByteBuffer.wrap(traceBytes);
        traceBb.order(ByteOrder.LITTLE_ENDIAN);
        var traceBbLong = traceBb.asLongBuffer();
        var addressSetList = new ArrayList<AddressSetView>(traceBytesLen / 16);

        for (int i = 0; i < traceBytesLen / 16; i++) {
            var start = traceBbLong.get(i * 2);
            var end = traceBbLong.get(i * 2 + 1);
            var start_addr = baseAddress.add(start);
            var end_addr = baseAddress.add(end);
            var as = new AddressSet(start_addr, end_addr);
            addressSetList.add(as);
        }

        return addressSetList;
    }
}
