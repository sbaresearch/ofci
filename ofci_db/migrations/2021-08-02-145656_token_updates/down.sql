ALTER TABLE functions
ALTER COLUMN tokens TYPE BYTEA;

ALTER TABLE functions
DROP COLUMN token_count;
