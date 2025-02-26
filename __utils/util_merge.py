import pymupdf
from typing import List

# Define the list of PDF files to merge
file_names: List[str] = []
for i in range(1, 23):
    file_names.append(f"../USC-CSCI699/Lecture {i}.pdf")

# Create a new PDF document as the merge target
merged_pdf = pymupdf.open()

for file_name in file_names:
    try:
        source_pdf = pymupdf.open(file_name)
        merged_pdf.insert_pdf(source_pdf)
        source_pdf.close()
        print(f"已合并: {file_name}")
    except Exception as e:
        print(f"合并 {file_name} 时出错: {str(e)}")

# Ensure the merged PDF has an even number of pages
if merged_pdf.page_count % 2 == 1:
    # Get the dimensions of the first page as the size of the new page
    if merged_pdf.page_count > 0:
        first_page = merged_pdf.load_page(0)
        width, height = first_page.bound().width, first_page.bound().height
    else:
        # If there are no pages, use the A4 size
        width, height = 595, 842

    # Add a blank page
    merged_pdf.new_page(width=width, height=height)
    print("已添加一个空白页以确保页数为偶数")

merged_pdf.save("notes.pdf")
merged_pdf.close()

print("所有PDF已成功合并为 notes.pdf")
