import argparse
import subprocess
import sys
import glob
import os

SUBTITLE_EXTS = ("*.srt", "*.txt", "*.ass", "*.ssa", "*.vtt", "*.sub", "*.lrc", "*.md")


def join_with_paragraphs(lines, max_len=500):
    """Join lines into paragraphs, inserting a blank line after roughly max_len characters."""
    paragraphs = []
    current = []
    current_len = 0
    for line in lines:
        text = line.strip()
        if not text:
            continue
        current.append(text)
        current_len += len(text) + 1
        if current_len >= max_len:
            paragraphs.append(" ".join(current))
            current = []
            current_len = 0
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def validate_blank_lines(lines, filename):
    """Detect root-cause locations where the 8-line blank pattern shifts.

    Expected pattern: every 8 lines, lines 4 and 8 (1-indexed within group) are blank.
    When extra or missing lines cause a shift, this function traces back to the
    root-cause location of each shift rather than reporting every subsequent mismatch.
    """
    n = len(lines)
    offset = 0  # cumulative shift from insertions/deletions
    errors = []
    i = 0

    while i < n:
        expected_pos = (i - offset) % 8
        is_blank = lines[i].strip() == ""
        should_be_blank = expected_pos in (3, 7)

        if is_blank == should_be_blank:
            i += 1
            continue

        # Mismatch at line i. Try shift deltas [-4, +4] to find the best explanation.
        best_delta = None
        best_score = -1
        look_ahead = min(32, n - i)

        for delta in range(-4, 5):
            if delta == 0:
                continue
            score = 0
            for j in range(look_ahead):
                pos = (i + j - offset - delta) % 8
                jblank = lines[i + j].strip() == ""
                jshould = pos in (3, 7)
                if jblank == jshould:
                    score += 1
            if score > best_score:
                best_score = score
                best_delta = delta

        line_num = i + 1
        new_offset = offset + (best_delta or 0)
        if best_delta is not None and best_delta > 0:
            errors.append(
                f"  Line {line_num}: {best_delta} extra line(s) inserted here "
                f"(cumulative offset: {offset:+d} -> {new_offset:+d})"
            )
        elif best_delta is not None and best_delta < 0:
            errors.append(
                f"  Line {line_num}: {-best_delta} line(s) missing here "
                f"(cumulative offset: {offset:+d} -> {new_offset:+d})"
            )
        else:
            errors.append(f"  Line {line_num}: pattern break (cannot determine shift)")

        # Show context: 2 lines before and after the problematic line
        ctx_start = max(0, i - 2)
        ctx_end = min(n, i + 3)
        for ci in range(ctx_start, ctx_end):
            marker = ">>>" if ci == i else "   "
            content = lines[ci].rstrip()
            errors.append(f"    {marker} line {ci + 1}: {content!r}")
        errors.append("")

        if best_delta is not None:
            offset = new_offset
        i += 1

    if errors:
        print(f"Blank line validation failed for {filename}:")
        print("Expected pattern: every 8 lines, the 4th and 8th are blank.\n")
        for msg in errors:
            print(msg)
        sys.exit(1)


def process_file(filename, paragraph_len=500):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    validate_blank_lines(lines, filename)

    lines_english = lines[2::8]
    lines_chinese = lines[6::8]

    lines_english_new = join_with_paragraphs(lines_english, paragraph_len)
    lines_chinese_new = join_with_paragraphs(lines_chinese, paragraph_len)

    base, ext = os.path.splitext(filename)
    outfile = base + ".md"

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(
            "## English\n\n"
            + lines_english_new
            + "\n\n## 中文\n\n"
            + lines_chinese_new
            + "\n"
        )

    # If the source file is not the output .md itself, delete the original
    if os.path.abspath(filename) != os.path.abspath(outfile):
        os.remove(filename)

    print(f"Done: {filename} -> {outfile}")


def select_with_fzf():
    files = sorted({f for ext in SUBTITLE_EXTS for f in glob.glob(ext)})
    if not files:
        print("No subtitle files found in the current directory.")
        print(f"Supported extensions: {', '.join(SUBTITLE_EXTS)}")
        sys.exit(1)

    try:
        result = subprocess.run(
            ["fzf", "--prompt=Select a subtitle file> ", "--height=40%", "--reverse"],
            input="\n".join(files),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            "fzf is not installed. Please install it or use: python parse.py -f <file>"
        )
        sys.exit(1)

    chosen = result.stdout.strip()
    if not chosen:
        print("No file selected.")
        sys.exit(0)

    return chosen


def main():
    parser = argparse.ArgumentParser(description="Parse bilingual subtitle files.")
    parser.add_argument("-f", "--file", help="Subtitle file to process")
    parser.add_argument(
        "-p",
        "--paragraph-len",
        type=int,
        default=500,
        help="Approximate max character count per paragraph (default: 500)",
    )
    args = parser.parse_args()

    if args.file:
        if not os.path.isfile(args.file):
            print(f"File not found: {args.file}")
            sys.exit(1)
        process_file(args.file, args.paragraph_len)
    else:
        chosen = select_with_fzf()
        process_file(chosen, args.paragraph_len)


if __name__ == "__main__":
    main()
