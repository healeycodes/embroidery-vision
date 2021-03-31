import approx_threads
import unittest


class TestApproxThreads(unittest.TestCase):
    def test_end_to_end(self):
        pomegranate = 'test/pomegranate.jpg'
        circle_image, original_image = approx_threads.get_largest_circle(pomegranate, debug=False)
        thread_palette = approx_threads.generate_palette(circle_image, original_image, debug=False)

        self.assertIn('#3857 Rosewood Dark', thread_palette)

        computer = 'test/computer.jpg'
        circle_image, original_image = approx_threads.get_largest_circle(computer, debug=False)
        thread_palette = approx_threads.generate_palette(circle_image, original_image, debug=False)

        self.assertIn('#156 Blue Violet Med Lt', thread_palette)
