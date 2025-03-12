import fitz
import os
from PIL import Image

def preprocess_latex(latex_code):
    latex_code = latex_code.strip()
    latex_code = latex_code.replace(r"\documentclass{article}", r"\documentclass[border=10pt]{standalone}")
    latex_code = latex_code.replace(r"\begin{center}", r"")
    latex_code = latex_code.replace(r"\end{center}", r"")
    latex_code = latex_code.replace(")\n\\draw", ");\n\\draw")
    return latex_code

def pdf2jpg(pdfPath, imgPath, zoom_x, zoom_y, rotation_angle=0, reconvert=False):

    try:
        # Open the PDF file
        pdf = fitz.open(pdfPath)
        if os.path.exists(imgPath) and not reconvert:
            print(f"Image file {imgPath} already exists, skip converting...")
            return True
        assert pdf.page_count > 0, "PDF page count is 0"
        # for pg in range(0, pdf.pageCount):
        page = pdf[0]
        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotation_angle)
        pm = page.get_pixmap(matrix=trans, alpha=False)
        print(pm.width, pm.height)
        pm._writeIMG(imgPath, format_="jpg", jpg_quality=100)
        pdf.close()
        image = Image.open(imgPath)

        width, height = image.size
        print(f"PDF file {pdfPath} has been successfully converted to a JPEG image, saved in {imgPath}. Height: {height} pixels, Width: {width} pixels")
        return True
    except Exception as e:
        print(f"Failed to convert PDF file {pdfPath} to JPEG image, due to the following error:\n{e}\n")
        return False

def compile_latex(folder, file_name, latex_code):
    succ = False
    if not os.path.exists(folder):
        print(f"if exists <folder>: {os.path.exists(folder)}")
        os.makedirs(folder, exist_ok=True)

    with open(f"{folder}/{file_name}.tex", "w") as f:
        f.write(latex_code)
    try:
        exit_code = os.system(f"pdflatex -interaction=batchmode -output-directory={folder} {folder}/{file_name}.tex")
        if exit_code == 0 and os.path.exists(f"{folder}/{file_name}.pdf"):
            print("Successfully compiled!")
            succ = True
        else:
            exit_code = os.system(f"xelatex -interaction=batchmode -output-directory={folder} {folder}/{file_name}.tex")
            if exit_code == 0 and os.path.exists(f"{folder}/{file_name}.pdf"):
                print("Successfully compiled!")
                succ = True
            else:
                print(f"Failed to compile. latex file: {folder}/{file_name}.tex")
                # delete failed
    except Exception as e:
        print(e)

    return succ

if __name__ == "__main__":
    path = "test.pdf"
    save_path = path.replace(".pdf", ".jpg")
    pdf2jpg(pdfPath=path, imgPath=save_path, zoom_x=1, zoom_y=1, rotation_angle=0)
