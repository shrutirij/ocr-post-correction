"""Script to get a first pass OCR output from the Google Vision OCR system on a set of scanned images.

Copyright (c) 2021, Shruti Rijhwani
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


import argparse
import io
import re
import glob
import os.path
import json
from collections import defaultdict
from google.cloud import vision
from google.protobuf.json_format import MessageToJson


class OCR:
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def detect_text(self, image_path):
        with io.open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = self.client.document_text_detection(image=image)
        return response

    def return_full_text(self, img_path):
        response = self.detect_text(img_path)
        return response.full_text_annotation.text

    def return_json(self, img_path):
        response = self.detect_text(img_path)
        blocks = {}
        count = 0
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                cur_block = {}
                cur_block["bounding_box"] = []
                for vertex in block.bounding_box.vertices:
                    cur_block["bounding_box"].append((vertex.x, vertex.y))
                cur_block["lang"] = []
                for lang in block.property.detected_languages:
                    cur_block["lang"].append((lang.language_code))
                cur_block["text"] = ""
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            cur_block["text"] += symbol.text
                            break_type = symbol.property.detected_break.type
                            if break_type == 1 or break_type == 2:
                                cur_block["text"] += " "
                            elif break_type == 3 or break_type == 5:
                                cur_block["text"] += "\n"
                            elif break_type == 4:
                                cur_block["text"] += "-\n"
                blocks[count] = cur_block
                count += 1

        blocks["fulltext"] = response.full_text_annotation.text
        return blocks


def get_ocr(image_paths, json_out):
    responses = []
    ocr_client = OCR()
    for img in image_paths:
        print(img)
        if json_out:
            responses.append(ocr_client.return_json(img))
        else:
            responses.append(ocr_client.return_full_text(img))
    return responses


def write_outputs(image_paths, ocr_responses, output_folder, json_out):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for img, ocr in zip(image_paths, ocr_responses):
        filename, _ = os.path.splitext(img.split("/")[-1])
        if json_out:
            with open("".join([output_folder, "/", filename, ".json"]), "w") as out:
                json.dump(
                    ocr, out, separators=(",", ": "), ensure_ascii=False, indent=4
                )
        else:
            with open("".join([output_folder, "/", filename, ".txt"]), "w") as out:
                out.write(ocr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path of a single input image for processing.")
    parser.add_argument(
        "--image_folder",
        help="Path of a folder that contains many images for processing.",
    )
    parser.add_argument("--output_folder", help="Output folder for the OCR text.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Enable for JSON output format which includes OCR text and bounding boxes. Default is plain text format.",
    )
    args = parser.parse_args()

    if args.image_folder:
        image_paths = sorted(glob.glob(args.image_folder + "/*.png"))
    else:
        image_paths = [args.image]

    ocr_responses = get_ocr(image_paths, args.json)
    write_outputs(image_paths, ocr_responses, args.output_folder, args.json)
