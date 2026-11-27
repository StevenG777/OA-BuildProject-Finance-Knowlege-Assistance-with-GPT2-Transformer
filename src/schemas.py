"""JSON schema constrains structure, not financial truth."""
GENERATION_SCHEMA = {
    "type": "object", 
    "additionalProperties": False,
    "properties": {
        "generation_status": {
            "type": "string", 
            "enum": ["generated", "skipped"]
        },

        "skip_reason": {
            "type": ["string", "null"]
        },

        "grounding": {
            "type": "string", 
            "enum": ["directly_supported", "derived_from_source", "insufficient_support"]
        },

        "source_evidence": {
            "type": ["string", "null"]
        },

        "question": {
            "type": ["string", "null"]
        },

        "answer": {
            "type": ["string", "null"]
        },

        "formula_latex": {
            "type": ["string", "null"]
        },
    },
    "required": [
        "generation_status", 
        "skip_reason", 
        "grounding",
        "source_evidence", 
        "question", 
        "answer", 
        "formula_latex"
    ],
}


GENERATION_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "finance_item",
        "schema": GENERATION_SCHEMA,
        "strict": True
    }
}