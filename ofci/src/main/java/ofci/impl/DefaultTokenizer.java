package ofci.impl;

import ofci.Tokenizer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class DefaultTokenizer implements Tokenizer {
    private final StringBuilder content;
    private final ByteBuffer bbuff;

    public DefaultTokenizer(int stringBuilderSize) {
        content = new StringBuilder(stringBuilderSize);
        bbuff = ByteBuffer.allocate(8);
        bbuff.order(ByteOrder.LITTLE_ENDIAN);
    }

    public DefaultTokenizer() {
        this(32768);
    }

    public void addChar(Character c) {
        content.append(' ');
        content.append(c);
    }

    public void addString(String s) {
        content.append(' ');
        content.append(s);
    }

    public void addLong(long value, int numBytes) {
        bbuff.clear();
        bbuff.putLong(value);
        var a = bbuff.array();

        // Find first non-null byte
        for (; numBytes > 1; numBytes--) {
            if (a[numBytes - 1] != 0)
                break;
        }

        for (int j = 0; j < numBytes; j++)
            content.append(String.format(" %02x", a[j]));
    }

    public void clearState() {
        content.setLength(0);
    }

    public String getBufferContents() {
        return content.toString();
    }
}
