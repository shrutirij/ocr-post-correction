"""[summary]

Author: Shruti Rijhwani
Contact: srijhwan@cs.cmu.edu

Please cite:
OCR Post Correction for Endangered Language Texts (EMNLP 2020)
https://www.aclweb.org/anthology/2020.emnlp-main.478/
"""

from pdf2image import convert_from_path
import argparse
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf")
    parser.add_argument("--pdf_folder")
    parser.add_argument("--output_folder")
    args = parser.parse_args()
    if args.pdf_folder:
        pdfs = glob.glob(args.pdf_folder + "/*.pdf")
    else:
        pdfs = [args.pdf]

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    images = []

    for pdf in pdfs:
        images += convert_from_path(pdf, dpi=400, output_folder=args.output_folder)

    for i, page in enumerate(images):
        print(i)
        page.save("{}/{}.png".format(args.output_folder, i), "PNG")

    temp_files = glob.glob("{}/*.ppm".format(args.output_folder))
    for f in temp_files:
        os.remove(f)
