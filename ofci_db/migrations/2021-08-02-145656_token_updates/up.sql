-- Change column type to text, because we're just using json
ALTER TABLE functions
ALTER COLUMN tokens TYPE TEXT;

-- Add a token count
ALTER TABLE functions
ADD token_count INTEGER;
