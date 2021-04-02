import argparse
from approx_dmc_threads import get_largest_circle, generate_palette

parser = argparse.ArgumentParser(description='Get a DMC color palette for an image of an embroidery hoop.')
parser.add_argument('input', metavar='file.jpg', help='an input file.')
parser.add_argument('--output', default=False, help='an output file.',
                    required=False, metavar='file.jpg')
parser.add_argument('--debug', action='store_true', help='set debug mode. Saves debug files to current directory.')
args = parser.parse_args()

circle_image, original_image = get_largest_circle(args.input, debug=args.debug)
thread_palette = generate_palette(circle_image, original_image, output_file=args.output, debug=args.debug)

print(thread_palette)
