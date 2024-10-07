#!/bin/bash

# Specify the docs directory
DOCS_DIR="docs/"

# Iterate over all .md files in the docs directory and subdirectories
find "$DOCS_DIR" -name "*.md" | while read -r file; do
    # Convert the file to .rst using pandoc
    pandoc -s "$file" -t rst -o "${file%.md}.rst"
    
    # Check if the conversion was successful before deleting
    if [ $? -eq 0 ]; then
        # Delete the original .md file
        rm "$file"
        echo "Converted and removed: $file"
    else
        echo "Failed to convert: $file"
    fi
done