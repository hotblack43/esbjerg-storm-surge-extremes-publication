#!/bin/bash

# dedup_safe.sh: Find and optionally remove duplicate files (by content hash)
# Only operates on files in the top level of the given directory (non-recursive)

TARGET_DIR="${1:-.}"  # defaults to current directory

echo "🔍 Scanning for duplicate files in: $TARGET_DIR"
echo "(Only files directly inside this folder are considered — subdirectories ignored)"

# Step 1: Compute SHA256 checksums (non-recursive)
find "$TARGET_DIR" -maxdepth 1 -type f -exec sha256sum "{}" + | sort > checksums.txt

# Step 2: Group files by checksum and find duplicates
awk '
{
    hash = $1
    file = substr($0, index($0,$2))
    files[hash] = (hash in files) ? files[hash] "\n" file : file
}
END {
    for (h in files) {
        split(files[h], arr, "\n")
        if (length(arr) > 1) {
            print "## Duplicate group"
            for (i in arr) print arr[i]
            print ""
        }
    }
}' checksums.txt > duplicate_groups.txt

# Step 3: Build list of files to delete (keep only one in each group)
> to_delete.txt  # start with empty file
awk '
BEGIN { keep = "" }
/^## Duplicate group/ {
    keep = "";  # reset for each group
    next
}
NF {
    if (keep == "") {
        keep = $0
        print "Keeping: " keep
    } else {
        print "To delete: " $0
        print $0 >> "to_delete.txt"
    }
}
' duplicate_groups.txt

# Step 4: Preview deletion list
echo
echo "🧾 Files that would be deleted:"
cat to_delete.txt
echo

# Step 5: Confirm deletion
read -p "❓ Do you want to delete the listed duplicates? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "🗑 Deleting..."
    xargs -d '\n' -a to_delete.txt rm -v
    echo "✅ Done."
else
    echo "🚫 No files were deleted. See 'to_delete.txt' to review manually."
fi

# Cleanup
rm checksums.txt duplicate_groups.txt 2>/dev/null

