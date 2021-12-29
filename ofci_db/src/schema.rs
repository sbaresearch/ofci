table! {
    category_names (id) {
        id -> Int4,
        name -> Varchar,
    }
}

table! {
    function_names (id) {
        id -> Int4,
        name -> Varchar,
    }
}

table! {
    functions (id) {
        id -> Int4,
        name -> Int4,
        project_group -> Int4,
        tool -> Int4,
        category -> Int4,
        call_count -> Int4,
        hash -> Uuid,
        contents -> Text,
        tokens -> Nullable<Text>,
        token_count -> Nullable<Int4>,
    }
}

table! {
    group_names (id) {
        id -> Int4,
        name -> Varchar,
    }
}

table! {
    tool_names (id) {
        id -> Int4,
        name -> Varchar,
    }
}

joinable!(functions -> category_names (category));
joinable!(functions -> function_names (name));
joinable!(functions -> group_names (project_group));
joinable!(functions -> tool_names (tool));

allow_tables_to_appear_in_same_query!(
    category_names,
    function_names,
    functions,
    group_names,
    tool_names,
);
