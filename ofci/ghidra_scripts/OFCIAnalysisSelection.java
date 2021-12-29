import ghidra.app.util.headless.HeadlessScript;

public class OFCIAnalysisSelection extends HeadlessScript {
    @Override
    public void run() throws Exception {
        setAnalysisOption(currentProgram, "Apply Data Archives", "false");
        //setAnalysisOption(currentProgram, "Call Convention ID", "false");
        setAnalysisOption(currentProgram, "Call-Fixup Installer", "false");
        //setAnalysisOption(currentProgram, "Create Address Tables", "false");
        //setAnalysisOption(currentProgram, "Create Address Tables - One Time", "false");
        //setAnalysisOption(currentProgram, "Create Function", "false");
        //setAnalysisOption(currentProgram, "DWARF", "false");
        //setAnalysisOption(currentProgram, "Data Reference", "false");
        setAnalysisOption(currentProgram, "Demangler GNU", "false");
        //setAnalysisOption(currentProgram, "Disassemble Entry Points", "false");
        setAnalysisOption(currentProgram, "Embedded Media", "false");
        setAnalysisOption(currentProgram, "External Entry References", "false");
        setAnalysisOption(currentProgram, "Function Start Search", "false");
        setAnalysisOption(currentProgram, "Function Start Search After Code", "false");
        setAnalysisOption(currentProgram, "Function Start Search After Data", "false");
        setAnalysisOption(currentProgram, "Shared Return Calls", "false");
        //setAnalysisOption(currentProgram, "Subroutine References", "false");
        setAnalysisOption(currentProgram, "ASCII Strings", "false");
        // NOTE: This takes the longest, but actually helps us find more instructions..
        // setAnalysisOption(currentProgram, "Decompiler Switch Analysis", "false");
        setAnalysisOption(currentProgram, "ELF Scalar Operand References", "false");
        setAnalysisOption(currentProgram, "Function ID", "false");
        setAnalysisOption(currentProgram, "GCC Exception Handlers", "false");
        setAnalysisOption(currentProgram, "Stack", "false");
        setAnalysisOption(currentProgram, "Reference", "false");
        setAnalysisOption(currentProgram, "Non-Returning Functions - Discovered", "false");
        setAnalysisOption(currentProgram, "Non-Returning Functions - Known", "false");
        setAnalysisOption(currentProgram, "x86 Constant Reference Analyzer", "false");
    }
}
