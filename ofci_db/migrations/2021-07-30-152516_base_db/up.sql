CREATE TABLE function_names (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE
);

CREATE TABLE group_names (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE
);

CREATE TABLE category_names (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE
);

CREATE TABLE tool_names (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE
);

CREATE TABLE functions (
    id SERIAL PRIMARY KEY,
    name INTEGER NOT NULL REFERENCES function_names,
    project_group INTEGER NOT NULL REFERENCES group_names,
    tool INTEGER NOT NULL REFERENCES tool_names,
    category INTEGER NOT NULL REFERENCES category_names,
    call_count INTEGER NOT NULL,
    hash UUID NOT NULL,
    contents TEXT NOT NULL,
    tokens BYTEA -- Can be null, tokens are generated later on
);
