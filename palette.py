from functools import lru_cache
import sys
import cv2
import numpy as np
from collections import Counter
from nearest_dmc import rgb_to_dmc, dmc_colors

class CircleNotFound(Exception):
    pass

def get_largest_circle(file_path, debug=False):
    original_img = cv2.imread(file_path) 

    # scale down
    maxwidth, maxheight = 400, 400
    f1 = maxwidth / original_img.shape[1]
    f2 = maxheight / original_img.shape[0]
    f = min(f1, f2)
    dim = (int(original_img.shape[1] * f), int(original_img.shape[0] * f))
    img = cv2.resize(original_img, dim)

    # apply destructive filters
    filtered_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.GaussianBlur(filtered_img, (5, 5), 0)
    filtered_img = cv2.medianBlur(filtered_img, 5)
    filtered_img = cv2.adaptiveThreshold(filtered_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5)
    kernel = np.ones((5, 5), np.uint8)
    filtered_img = cv2.erode(filtered_img, kernel, iterations=1)
    filtered_img = cv2.dilate(filtered_img, kernel, iterations=1)

    circles = cv2.HoughCircles(filtered_img, cv2.HOUGH_GRADIENT, 1, 200, param1=30, param2=45, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = list(np.round(circles[0, :]).astype('int'))

        for idx, t in enumerate(circles):
            x, y, r = circles[idx]
            circles[idx] = (
                int(x / f),
                int(y / f),
                # can shrink the radius to trim the hoop's border here
                int((r / f) * 1.0)
            )

        # valid circles must have about half of their volume in the picture
        # check that the center point of the circle is at least half a radius
        # away from any of the edges of the image
        valid_circles = []
        for circle in circles:
            x, y, r = circle
            h, w, _ = img.shape
            if x - r < r / 2:
                continue
            if w - x + r < r:
                continue
            if y - r < r / 2:
                continue
            if h - x + r < r:
                continue
            valid_circles.append(circle)
       
        valid_circles.sort(key=lambda circle: circle[2])
        largest_circle = circles.pop()

        if debug:
            # draw green around the largest circle
            # draw red around the other circles
            for _, t in enumerate(valid_circles):
                x, y, r = t
                cv2.circle(original_img, (x, y), r, (0, 0, 255), 2)
                cv2.rectangle(original_img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
  
            large_x, large_y, large_r = largest_circle
            cv2.circle(original_img, (large_x, large_y), large_r, (0, 255, 0), 2)
            cv2.rectangle(original_img, (large_x - 5, large_y - 5), (large_x + 5, large_y + 5), (0, 128, 255), -1)

            cv2.imshow('filtered_img', filtered_img)
            cv2.imshow('img', original_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return largest_circle, original_img

    else:
        raise CircleNotFound()

def generate_palette(circle_image, original_img, save=False):
    x, y, r = circle_image

    div = 64 # reduce color space
    reduced_img = original_img // div * div + div // 2
    reduced_img = original_img
    
    # https://stackoverflow.com/a/36054249
    mask = np.zeros(reduced_img.shape, np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    where = np.where(mask == 255)
    intensity_values_from_original = reduced_img[where[0], where[1]]
    palette = []

    @lru_cache
    # since we limit the color space, cache k-d tree lookups
    def cached_rgb_to_dmc(r, g, b):
        return rgb_to_dmc(r, g, b)

    for _, color in enumerate(intensity_values_from_original):
        palette.append(
            # don't forget bgr -> rgb
            cached_rgb_to_dmc(color[2], color[1], color[0])['index']
        )

    c = Counter(palette)
    with_percentage = [(i, c[i] / len(palette) * 100.0) for i, _ in c.most_common()]

    # limit low occuring threads
    filtered = [color for color in with_percentage if color[1] > 1]
    with_dmc = [
        f"#{dmc_colors[color[0]]['floss']} {dmc_colors[color[0]]['description']}"
            for color in filtered]

    if save:
        cv2.circle(reduced_img, (x, y), r, (0, 255, 0), 2)
        _, w, _ = reduced_img.shape
        size = int(w / len(filtered))
        y = size
        x = 0
        for idx, color in enumerate(filtered):
            b, g, r = dmc_colors[color[0]]['blue'], dmc_colors[color[0]]['green'], dmc_colors[color[0]]['red']
            cv2.rectangle(reduced_img, (size * idx, 0), ((size * idx) + size, size), (b, g, r), -1)
            cv2.putText(reduced_img, '#' + dmc_colors[color[0]]['floss'], (size * idx, size-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
        cv2.imshow('img', reduced_img)
        cv2.imwrite('pc.jpg', reduced_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return with_dmc


circle_image, original_img = get_largest_circle(sys.argv[1], debug=False)
thread_palette = generate_palette(circle_image, original_img, save=True)
print(thread_palette)
