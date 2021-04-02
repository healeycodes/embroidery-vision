from collections import Counter
from functools import lru_cache
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from nearest_dmc import rgb_to_dmc, dmc_colors

MAXWIDTH, MAXHEIGHT = 400, 400

class CircleNotFound(Exception):
    pass

def apply_destructive_filters(image, debug=False):
    '''
    Apply a series of destructive filters with the aim of making circles more
    visible to the HoughCircles function.
    '''
    kernel = np.ones((5, 5), np.uint8)

    filtered_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        debug_image = filtered_image
    filtered_image = cv2.GaussianBlur(filtered_image, (5, 5), 0)
    if debug:
        debug_image = cv2.hconcat([debug_image, filtered_image])
    filtered_image = cv2.medianBlur(filtered_image, 5)
    if debug:
        debug_image = cv2.hconcat([debug_image, filtered_image])
    filtered_image = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5)
    if debug:
        debug_image = cv2.hconcat([debug_image, filtered_image])
    filtered_image = cv2.erode(filtered_image, kernel, iterations=1)
    if debug:
        debug_image = cv2.hconcat([debug_image, filtered_image])
    filtered_image = cv2.dilate(filtered_image, kernel, iterations=1)
    if debug:
        debug_image = cv2.hconcat([debug_image, filtered_image])
        cv2.imwrite('debug_destructive_filters.jpg', debug_image)
    return filtered_image

def get_scaled_down_image(image):
    '''
    Use this file's size constants to scale down an image.
    '''
    w = MAXWIDTH / image.shape[1]
    h = MAXHEIGHT / image.shape[0]
    scale = min(w, h)
    dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    return scale, cv2.resize(image, dim)

def get_circles(original_image, filtered_image, scale):
    '''
    Find circle areas that are most likely to be embroidery hoops. Circles
    are ordered by size and circles which are mostly of screen are discarded.
    '''
    small_circles = cv2.HoughCircles(filtered_image, cv2.HOUGH_GRADIENT, 1, 200, param1=30, param2=45, minRadius=0, maxRadius=0)
    scaled_up_circles = []
    for small_circle in np.round(small_circles[0, :]).astype('int'):
        x, y, r = small_circle
        # shrink the radius to trim the hoop's border
        shrink_amount = 0.99
        scaled_up_circles.append((
            int(x / scale),
            int(y / scale),
            int((r / scale) * shrink_amount)
        ))

    # valid circles must have about half of their volume in the picture
    # check that the center point of the circle is at least half a radius
    # away from any of the edges of the image
    valid_circles = []
    for circle in scaled_up_circles:
        h, w, _ = original_image.shape
        x, y, r = circle
        if x + r / 2 > w or x - r / 2 < 0:
            continue
        if y + r / 2 > h or y - r / 2 < 0:
            continue
        valid_circles.append(circle)

    if len(valid_circles) == 0:
        raise CircleNotFound()
    
    valid_circles.sort(key=lambda circle: circle[2])
    return valid_circles

def save_debug_circles(debug_filtered_image, valid_circles, largest_circle):
    '''
    Display the circles that were found. Draw green around the largest circle.
    Draw red around the other circles.
    '''
    for _, t in enumerate(valid_circles):
        x, y, r = t
        cv2.circle(debug_filtered_image, (x, y), r, (0, 0, 255), 10)
        cv2.rectangle(debug_filtered_image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    large_x, large_y, large_r = largest_circle
    cv2.circle(debug_filtered_image, (large_x, large_y), large_r, (0, 255, 0), 10)
    cv2.rectangle(debug_filtered_image, (large_x - 5, large_y - 5), (large_x + 5, large_y + 5), (0, 255, 0), -1)
    cv2.imwrite('debug_filtered_image.jpg', debug_filtered_image)

def get_largest_circle(file_path, debug=False):
    '''
    Find the largest valid circle (which is hopefully an embroidery hoop) for
    a given file path.
    '''
    original_image = cv2.imread(file_path) 

    scale, scaled_down_image = get_scaled_down_image(original_image)
    filtered_image = apply_destructive_filters(scaled_down_image, debug=debug)
    circles = get_circles(original_image, filtered_image, scale)
    largest_circle = circles.pop()

    if debug:
        debug_filtered_image = cv2.cvtColor(filtered_image.copy(), cv2.COLOR_GRAY2BGR)
        dim = (int(filtered_image.shape[1] / scale), int(filtered_image.shape[0] / scale))
        save_debug_circles(cv2.resize(debug_filtered_image, dim), circles, largest_circle)
    
    return largest_circle, original_image

# https://stackoverflow.com/a/20715062
def quantize_image(image, div=64):
    '''
    Reduces the number of distinct colors used in an image.
    '''
    quantized = image // div * div + div // 2
    return quantized

def generate_palette(circle_image, original_image, output_file=False):
    '''
    Generate a color palette of DMC threads for a circle mask and image.
    Overlay a palette graphic on the image and return a breakdown of the
    DMC threads.
    '''
    x, y, r = circle_image

    # reduce the number of distinct colors in an image
    # while preserving the color appearance of the image as much as possible
    reduced_color_image = quantize_image(original_image)
    
    # we're only interested in the hoop area
    mask = np.zeros(reduced_color_image.shape, np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    where = np.where(mask == 255)
    circle_area = reduced_color_image[where[0], where[1]]

    # since we've limit the color space, cache k-d tree lookups
    @lru_cache
    def cached_rgb_to_dmc(r, g, b):
        return rgb_to_dmc(r, g, b)

    color_counter = Counter()
    for _, color in enumerate(circle_area):
        color_counter[
            # don't forget bgr -> rgb
            cached_rgb_to_dmc(color[2], color[1], color[0])['index']
        ] += 1
 
    # trim low occuring threads, generate text output
    with_percentage = [(i, color_counter[i] / len(circle_area) * 100.0) for i, _ in color_counter.most_common()]
    limit_low_occuring_threads = 2 # %
    filtered = [color for color in with_percentage if color[1] > limit_low_occuring_threads]
    with_dmc = [f"{dmc_colors[color[0]]['floss']} {dmc_colors[color[0]]['description']} {round(color[1], 2)}%" for color in filtered]

    # overlay the color palette on top of the image
    _, w, _ = original_image.shape
    size = int(w / len(filtered))
    y = size
    x = 0
    for idx, color in enumerate(filtered):
        b, g, r = dmc_colors[color[0]]['blue'], dmc_colors[color[0]]['green'], dmc_colors[color[0]]['red']
        cv2.rectangle(original_image, (size * idx, 0), ((size * idx) + size, size), (b, g, r), -1)
        cv2.putText(original_image, dmc_colors[color[0]]['floss'], (size * idx, size-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255-b, 255-g, 255-r), 1)

    if output_file:
        cv2.imwrite(output_file, original_image)

    return '\n'.join(with_dmc)
