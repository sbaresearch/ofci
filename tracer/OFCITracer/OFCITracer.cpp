#include <fstream>
#include "pin.H"

std::ofstream trace_file;
IMG main_binary;
ADDRINT base_address;

VOID Image(IMG img, VOID *v) {
    if (IMG_IsMainExecutable(img)) {
        main_binary = img;
        base_address = IMG_LowAddress(img);
    }
}

VOID StoreBBLRange(ADDRINT end_ptr, ADDRINT start_ptr) {
    ADDRINT start = start_ptr - base_address;
    ADDRINT end = end_ptr - base_address;
    trace_file.write(reinterpret_cast<char *>(&start), sizeof(ADDRINT));
    trace_file.write(reinterpret_cast<char *>(&end), sizeof(ADDRINT));
}

VOID Trace(TRACE trace, VOID* v) {
    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl)) {
        // Only trace instructions of the current main binary
        ADDRINT addr = BBL_Address(bbl);
        if (addr >= IMG_HighAddress(main_binary) || addr < IMG_LowAddress(main_binary))
            continue;

        INS_InsertCall(
            BBL_InsTail(bbl),
            IPOINT_BEFORE,
            AFUNPTR(StoreBBLRange),
            IARG_INST_PTR,
            IARG_ADDRINT,
            addr,
            IARG_END
        );
    }
}

VOID Fini(INT32 code, VOID *v) {
    trace_file.close();
}

int main(int argc, char * argv[]) {
    PIN_Init(argc, argv);

    IMG_AddInstrumentFunction(Image, nullptr);
    TRACE_AddInstrumentFunction(Trace, nullptr);

    trace_file.open("./bbl_trace.out");

    PIN_StartProgram();
    return 0;
}
