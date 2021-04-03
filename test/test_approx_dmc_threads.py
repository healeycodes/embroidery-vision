from approx_dmc_threads import get_largest_circle, generate_palette
import unittest


class TestApproxThreads(unittest.TestCase):
    def test_end_to_end(self):
        pomegranate = "test/pomegranate.jpg"
        circle_image, original_image = get_largest_circle(pomegranate, debug=False)
        thread_palette = generate_palette(
            circle_image, original_image, output_file=False
        )

        self.assertIn("#3857 Rosewood Dark", thread_palette)

        computer = "test/computer.jpg"
        circle_image, original_image = get_largest_circle(computer, debug=False)
        thread_palette = generate_palette(circle_image, original_image)

        self.assertIn("#156 Blue Violet Med Lt", thread_palette)
