from functools import lru_cache
import sys
import cv2
import numpy as np
from collections import Counter
from nearest_dmc import rgb_to_dmc, dmc_colors

class CircleNotFound(Exception):
    pass

def apply_destructive_filters(image):
    kernel = np.ones((5, 5), np.uint8)

    filtered_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.GaussianBlur(filtered_image, (5, 5), 0)
    filtered_image = cv2.medianBlur(filtered_image, 5)
    filtered_image = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5)
    filtered_image = cv2.erode(filtered_image, kernel, iterations=1)
    filtered_image = cv2.dilate(filtered_image, kernel, iterations=1)
    return filtered_image

def get_scaled_down_image(image):
    maxwidth, maxheight = 400, 400
    w = maxwidth / image.shape[1]
    h = maxheight / image.shape[0]
    scale = min(w, h)
    dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    return scale, cv2.resize(image, dim)

def get_circles(original_image, filtered_image, scale):

    small_circles = cv2.HoughCircles(filtered_image, cv2.HOUGH_GRADIENT, 1, 200, param1=30, param2=45, minRadius=0, maxRadius=0)
    scaled_up_circles = []
    for small_circle in np.round(small_circles[0, :]).astype('int'):
        x, y, r = small_circle
        # can shrink the radius to trim the hoop's border here
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

def save_debug_circles(original_image, filtered_image, valid_circles, largest_circle):
    # draw green around the largest circle
    # draw red around the other circles
    for _, t in enumerate(valid_circles):
        x, y, r = t
        cv2.circle(original_image, (x, y), r, (0, 0, 255), 2)
        cv2.rectangle(original_image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    large_x, large_y, large_r = largest_circle
    cv2.circle(original_image, (large_x, large_y), large_r, (0, 255, 0), 3)
    cv2.rectangle(original_image, (large_x - 5, large_y - 5), (large_x + 5, large_y + 5), (0, 255, 0), -1)

    cv2.imwrite('debug_filtered_image.jpg', filtered_image)
    cv2.imwrite('debug_original_image.jpg', original_image)

def get_largest_circle(file_path, debug=False):
    original_image = cv2.imread(file_path) 

    scale, scaled_down_img = get_scaled_down_image(original_image)
    filtered_image = apply_destructive_filters(scaled_down_img)

    circles = get_circles(original_image, filtered_image, scale)
    if len(circles) == 0:
        raise CircleNotFound()
    largest_circle = circles.pop()

    if debug:
        save_debug_circles(original_image.copy(), filtered_image, circles, largest_circle)
    
    return largest_circle, original_image


def generate_palette(circle_image, original_image, debug=False):
    x, y, r = circle_image

    div = 64 # reduce color space
    reduced_color_image = original_image // div * div + div // 2
    
    # https://stackoverflow.com/a/36054249
    mask = np.zeros(reduced_color_image.shape, np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    where = np.where(mask == 255)
    intensity_values_from_original = reduced_color_image[where[0], where[1]]

    # since we limit the color space, cache k-d tree lookups
    @lru_cache
    def cached_rgb_to_dmc(r, g, b):
        return rgb_to_dmc(r, g, b)

    c = Counter()
    for _, color in enumerate(intensity_values_from_original):
        c[
            # don't forget bgr -> rgb
            cached_rgb_to_dmc(color[2], color[1], color[0])['index']
        ] += 1
 
    with_percentage = [(i, c[i] / len(intensity_values_from_original) * 100.0) for i, _ in c.most_common()]

    limit_low_occuring_threads = 3 # %
    filtered = [color for color in with_percentage if color[1] > limit_low_occuring_threads]
    with_dmc = [
        f"{dmc_colors[color[0]]['floss']} {dmc_colors[color[0]]['description']}"
            for color in filtered]

    if debug:
        cv2.circle(original_image, (x, y), r, (0, 255, 0), 2)

    _, w, _ = original_image.shape
    size = int(w / len(filtered))
    y = size
    x = 0
    for idx, color in enumerate(filtered):
        b, g, r = dmc_colors[color[0]]['blue'], dmc_colors[color[0]]['green'], dmc_colors[color[0]]['red']
        cv2.rectangle(original_image, (size * idx, 0), ((size * idx) + size, size), (b, g, r), -1)
        cv2.putText(original_image, dmc_colors[color[0]]['floss'], (size * idx, size-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255-b, 255-g, 255-r), 1)
    cv2.imwrite('out.jpg', original_image)

    return with_dmc

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Missing argument. Provide a path to an image file.\ne.g. `python this_script.py example.jpg`')
        quit()

    debug = sys.argv[-1] == 'debug' # e.g. `python this_script.py example.jpg debug`
    circle_image, original_image = get_largest_circle(sys.argv[1], debug=debug)
    thread_palette = generate_palette(circle_image, original_image, debug=debug)
    print('\n'.join(thread_palette))
