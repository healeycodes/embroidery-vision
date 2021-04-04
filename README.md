[![End to end image tests](https://github.com/healeycodes/embroidery-vision/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/healeycodes/embroidery-vision/actions/workflows/python-app.yml)

# 🧵 embroidery-vision

> My blog post: [Computer Vision and Embroidery](https://healeycodes.com/computer-vision-and-embroidery/)

<br>

A CLI for finding the approximate embroidery floss colours from an image of an embroidery hoop.

OpenCV is used to locate the hoop area then the color space is reduced and matched to the limited set of DMC colors (see `dmc.csv`). A color palette is generated with the DMC identification number attached to each color.

<img src="https://github.com/healeycodes/embroidery-vision/blob/main/examples/example_out.jpg" height="400">

In order to find the hoop area, a series of destructive filters are used — the image is converted to gray and then the following are applied: `GaussianBlur`, `medianBlur`, `adaptiveThreshold`, `erode`, `dilate`. As we see below, this makes the hoop more identifiable to the circle Hough transform.

<img src="https://github.com/healeycodes/embroidery-vision/blob/main/examples/example_destructive_filters.jpg">

Usually, multiple circles are found but the largest and most central one is chosen. This logic was chosen after reviewing ~100 of the latest posts to r/embroidery.

To find the colors, the area is quantized and then the nearest color is looked up in a cached k-d tree of DMC colors. The lower occuring colors are filtered out from the palette.

## Usage

Tested with Python 3.8.

`pip install -r requirements.txt`

```
$ python cli.py  -h
usage: cli.py [-h] [--output file.jpg] [--debug] file.jpg

Get a DMC color palette for an image of an embroidery hoop.

positional arguments:
  file.jpg           an input file.

optional arguments:
  -h, --help         show this help message and exit
  --output file.jpg  an output file.
  --debug            set debug mode. Saves debug files to current directory.
```

## Limitations

This program doesn't take lighting conditions into account and doesn't filter out the base material (it thinks the background is another thread) so the accuracy is low.

## Tests

`python -m unittest discover test/`

## License

MIT.
