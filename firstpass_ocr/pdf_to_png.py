"""Script to convert a scanned document in PDF format into a set of PNG images, one per page.

Copyright (c) 2021, Shruti Rijhwani
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


from pdf2image import convert_from_path
import argparse
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Path of a single PDF file to convert.")
    parser.add_argument(
        "--pdf_folder",
        help="Path for folder that contains multiple PDF files to convert.",
    )
    parser.add_argument("--output_folder", help="Output folder for PNG files.")
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
