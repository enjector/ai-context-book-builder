#!/bin/bash

# Configuration
OUTPUT_MD="_book_MyBook.md"
OUTPUT_PDF="_book_MyBook.pdf"

echo "Initializing Book Builder..."

# 1. Reset the output file
rm -f "$OUTPUT_MD"

# 2. Loop through numeric chapters (0-99) and Appendices (A-Z)
# This automatically finds 1.md, 2.md, A.md etc. and sorts them naturally.
for f in $(ls *.md | sort -v); do
  # Skip the output file itself and README if present
  if [[ "$f" == "$OUTPUT_MD" || "$f" == "README.md" ]]; then continue; fi
  
  echo "Merging Chapter: $f"
  cat "$f" >> "$OUTPUT_MD"
  
  # Add the separator you used in your original script
  echo $'\n---\n' >> "$OUTPUT_MD" 
done

# 3. Cleanup Citations
# Removes [cite] tags which are common in Gemini outputs
echo "Cleaning citations..."
sed -i 's/\[cite[^]]*\]//g' "$OUTPUT_MD"

# 4. Check dependencies
if ! command -v markdown-pdf &> /dev/null; then
    echo "Error: markdown-pdf not found. Please run: npm install -g markdown-pdf"
    exit 1
fi

# 5. Generate PDF
# Uses the settings from your original script (A4 format, OpenSSL fix)
echo "Generating PDF..."
OPENSSL_CONF=/dev/null markdown-pdf -f 'A4' "$OUTPUT_MD" -o "$OUTPUT_PDF"

echo "Success! Created $OUTPUT_PDF"
