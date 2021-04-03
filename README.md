[![End to end image tests](https://github.com/healeycodes/embroidery-vision/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/healeycodes/embroidery-vision/actions/workflows/python-app.yml)

# embroidery-vision

A CLI for finding the approximate embroidery floss colours from an image of an embroidery hoop.

OpenCV is used to locate the hoop area then the color space is reduce and matched to the limited set of DMC colors (see `dmc.csv`). A color palette is generated with the DMC identification number attached to each color.

<img src="https://github.com/healeycodes/embroidery-vision/blob/main/examples/example_out.jpg" height="400">

In order to find the hoop area, a series of destructive filters are used — the image is converted to gray and then the following are applied: `GaussianBlur`, `medianBlur`, `adaptiveThreshold`, `erode`, `dilate`. As we see below, this makes the hoop more clearly identifiable to the circle Hough transform.

Usually, multiple circles are found but the largest and most central one is chosen.

<img src="https://github.com/healeycodes/embroidery-vision/blob/main/examples/example_destructive_filters.jpg">
