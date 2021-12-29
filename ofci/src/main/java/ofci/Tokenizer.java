package ofci;

public interface Tokenizer extends ParserState {
    public void addChar(Character c);

    public void addString(String s);

    public void addLong(long l, int numBytes);

    public String getBufferContents();
}
