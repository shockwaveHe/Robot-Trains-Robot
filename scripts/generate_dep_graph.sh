#!/bin/bash
MODULE_NAME="toddleroid"

# Directory where the output file will be saved
OUTPUT_DIR="results"
mkdir -p $OUTPUT_DIR

# File name for the generated graph
OUTPUT_FILE="$OUTPUT_DIR/${MODULE_NAME}_deps.png"

# Generate the dependency graph
pydeps --max-bacon=2 --noshow --only "$MODULE_NAME" --rmprefix "$MODULE_NAME." -x "*utils*" -T png -o "$OUTPUT_FILE" $MODULE_NAME

echo "Dependency graph generated at $OUTPUT_FILE"
