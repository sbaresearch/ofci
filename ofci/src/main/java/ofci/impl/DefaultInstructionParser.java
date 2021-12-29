package ofci.impl;

import ofci.InstructionParser;
import ofci.Tokenizer;
import ofci.FunctionMetadata;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.GenericAddress;
import ghidra.program.model.lang.Register;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.scalar.Scalar;
import ghidra.util.Msg;

public abstract class DefaultInstructionParser implements InstructionParser {
    private static final int[] ACCEPTED_CHAR = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    protected final Tokenizer tokenizer;
    protected final FunctionMetadata functionMetadata;
    protected boolean isCall = false;
    private boolean isAddition = false;

    protected DefaultInstructionParser(
        Tokenizer tokenizer,
        FunctionMetadata functionMetadata
    ) {
        this.tokenizer = tokenizer;
        this.functionMetadata = functionMetadata;
    }

    protected void handleOpcode(Instruction instruction) {
        final var opcode = instruction.getMnemonicString();

        this.tokenizer.addString(opcode);

        switch (opcode) {
            case "SYSCALL":
                final var refMan = instruction.getProgram().getReferenceManager();
                final var addr = instruction.getAddress();
                final var refs = refMan.getReferencesFrom(addr);

                long targetAddress = 0xffff;
                if (refs != null && refs.length > 0)
                    targetAddress = refs[0].getToAddress().getOffset();

                this.tokenizer.addLong(targetAddress, 2);
                break;
            case "CALL":
                this.isCall = true;
                break;
        }
    }

    protected void handleOperand(Instruction instruction, int index) {
        final var opRepList = instruction.getDefaultOperandRepresentationList(index);
        for (var obj : opRepList) {
            if (obj instanceof Character) {
                this.handleCharacter((Character)obj);
            } else if (obj instanceof Register) {
                this.handleRegister((Register)obj);
            } else if (obj instanceof Scalar) {
                this.handleScalar((Scalar)obj);
            } else if (obj instanceof GenericAddress) {
                this.handleAddress((Address)obj);
            } else {
                Msg.error(this, "Unknown operand type: " + obj.getClass());
            }
        }
    }

    protected void handleCharacter(Character chr) {
        // NOTE: Maybe log discarded characters?
        if (ACCEPTED_CHAR[chr] == 1) {
            if (chr == '+') {
                isAddition = true;
            } else if (isAddition) {
                isAddition = false;
                tokenizer.addChar('+');
                tokenizer.addChar(chr);
            } else {
                isAddition = false;
                tokenizer.addChar(chr);
            }
        }
    }

    protected void handleRegister(Register register) {
        if (isAddition) {
            tokenizer.addChar('+');
            isAddition = false;
        }
        tokenizer.addString(register.getName());
    }

    protected void handleScalar(Scalar scalar) {
        final var val = scalar.getValue();
        if (val < 0) {
            tokenizer.addChar('-');
            isAddition = false;
            tokenizer.addLong(Math.abs(val), 8);
        } else {
            if (isAddition) {
                tokenizer.addChar('+');
                isAddition = false;
            }
            tokenizer.addLong(val, 8);
        }
    }

    protected void handleAddress(Address addr) {
        final var fnId = this.functionMetadata.getGlobalInfo().getFunctionNameId(addr);
        if (fnId == null) {
            if (this.isCall) {
                if (isAddition) {
                    tokenizer.addChar('+');
                    isAddition = false;
                }

                tokenizer.addLong(0, 8);
            } else {
                final var val = addr.getOffset();
                if (val < 0) {
                    tokenizer.addChar('-');
                    isAddition = false;
                    tokenizer.addLong(Math.abs(val), 8);
                } else {
                    if (isAddition) {
                        tokenizer.addChar('+');
                        isAddition = false;
                    }
                    tokenizer.addLong(val, 8);
                }
            }
        } else {
            if (isAddition) {
                tokenizer.addChar('+');
                isAddition = false;
            }

            this.tokenizer.addLong(fnId, 4);
            this.functionMetadata.incReferencedFuncCount();
        }
    }

    public final FunctionMetadata getFunctionMetadata() {
        return functionMetadata;
    }

    public final void clearState() {
        functionMetadata.clearState();
    }

    public final void addInstruction(Instruction instruction) {
        this.isCall = false;
        this.handleOpcode(instruction);

        final var numOperands = instruction.getNumOperands();
        for (int i = 0; i < numOperands; i++) {
            this.handleOperand(instruction, i);
            this.addOperandSeparator(instruction, i);
        }

        this.addInstructionSeparator();
    }

    protected abstract void addOperandSeparator(Instruction instruction, int index);

    protected abstract void addInstructionSeparator();
}
