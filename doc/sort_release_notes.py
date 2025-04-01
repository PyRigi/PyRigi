import sys
import re
from collections import defaultdict

# Define categories for users and developers
user_categories = {
    "Feature:": "New features",
    "Update:": "Updates",
    "Fix:": "Bug fixes",
    "Doc:": "Documentation",
}

developer_categories = {
    "Test:": "Testing",
    "Setup:": "Technical setup",
    "Code:": "Code changes",
    "Guide:": "Development guide",
    "Minor:": "Minor changes",
}

full_changelog = ""


def parse_line(line):
    # Regex to capture PR prefix, title, author(s), URL, and number
    pattern = r"\* (\w+): *(.+?) by (.+) in (https://github\.com/.+/pull/(\d+))"
    line = line.strip()
    match = re.match(pattern, line)
    if match:
        prefix = match.group(1) + ":"
        title = match.group(2)
        author = match.group(3)
        url = match.group(4)
        pr_number = int(match.group(5))
        return prefix, title, author, pr_number, url
    elif line.startswith("**Full Changelog**:"):
        global full_changelog
        full_changelog = line
    elif line and not line.startswith("#"):
        return "Not recognized:", line, 0, "", ""
    return None


def categorize_prs(pr_list):
    # Dictionaries to hold categorized PRs
    user_prs = defaultdict(list)
    developer_prs = defaultdict(list)
    unsorted_prs = []

    for prefix, title, author, pr_number, url in pr_list:
        if prefix in user_categories:
            user_prs[user_categories[prefix]].append((pr_number, title, url, author))
        elif prefix in developer_categories:
            developer_prs[developer_categories[prefix]].append(
                (pr_number, title, url, author)
            )
        else:
            unsorted_prs.append(
                (pr_number, prefix, title, url, author)
            )  # Collect unsorted PRs

    # Sort by PR number within each category
    for category in user_prs:
        user_prs[category].sort()

    for category in developer_prs:
        developer_prs[category].sort()

    # Sort unsorted PRs by their PR number
    unsorted_prs.sort()

    return user_prs, developer_prs, unsorted_prs


def format_section(prs, categories):
    # Format the sections for either users or developers
    result = []
    for category, category_name in categories.items():
        if prs.get(category_name):
            result.append(f"### {category_name}")
            for pr_number, title, url, author in prs[category_name]:
                result.append(f"* {title} by {author} in {url}")
            result.append("")  # Empty line after each category
    return "\n".join(result)


def format_unsorted_section(unsorted_prs):
    # Format the unsorted section
    if unsorted_prs:
        result = ["## !!!!!!Unsorted!!!!!!"]
        for pr_number, prefix, title, url, author in unsorted_prs:
            result.append(f"* {prefix} {title} by {author} in {url}")
        return "\n".join(result)
    return ""


def process_file(input_file):
    # Read the file content
    with open(input_file, "r") as f:
        lines = f.readlines()

    pr_list = []
    for line in lines:
        parsed = parse_line(line)
        if parsed:
            pr_list.append(parsed)

    # Categorize PRs
    user_prs, developer_prs, unsorted_prs = categorize_prs(pr_list)

    # Format the sections
    user_section = format_section(user_prs, user_categories)
    developer_section = format_section(developer_prs, developer_categories)
    unsorted_section = format_unsorted_section(unsorted_prs)

    # Final result
    result = f"""## What's Changed

{user_section}


## Development

{developer_section}

{unsorted_section}

{full_changelog}
"""

    return result


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    formatted_notes = process_file(input_file)

    # Write the formatted release notes to a new file
    output_file = f"formatted_{input_file}"
    with open(output_file, "w") as f:
        f.write(formatted_notes)

    print(f"Formatted release notes have been written to: {output_file}")


if __name__ == "__main__":
    main()
