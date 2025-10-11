filename = "subtitles.txt"


def process_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines_english = lines[2::8]
    lines_chinese = lines[6::8]

    with open(filename, "w", encoding="utf-8") as f:
        lines_english_new = " ".join(line.strip() for line in lines_english)
        lines_chinese_new = " ".join(line.strip() for line in lines_chinese)
        f.write("English:\n" + lines_english_new + "\n\n中文：\n" + lines_chinese_new)
