#!/bin/bash

input_file="$1"
temp_file=$(mktemp)

while read -r line; do
    image_path=$(echo "$line" | cut -d' ' -f1)
    seg_path=$(echo "$line" | cut -d' ' -f2)
    new_seg_path=${seg_path/SegmentationClassAug/SegmentationClass}
    echo "$image_path $new_seg_path" >> "$temp_file"
done < "$input_file"

# Replace the original file with the temp file
mv "$temp_file" "$input_file"