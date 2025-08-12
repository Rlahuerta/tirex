import unittest
from tirex.utils.trade import TrailingStopOrder


class TestTrailingStopOrder(unittest.TestCase):

    def test_initial_absolute_buy(self):
        order = TrailingStopOrder(size=5, trail_value=10, initial_price=100, trail_type='absolute')
        self.assertEqual(order.direction, 1)
        self.assertEqual(order.current_stop_price, 90)
        self.assertEqual(order.initial_price, 100)

    def test_initial_percentage_buy(self):
        order = TrailingStopOrder(size=3, trail_value=10, initial_price=200, trail_type='percentage')
        expected = 200 - 0.1 * 200
        self.assertAlmostEqual(order.current_stop_price, expected)

    def test_initial_absolute_sell(self):
        order = TrailingStopOrder(size=1, trail_value=-10, initial_price=100, trail_type='absolute')
        expected = 100 - (-10)
        self.assertEqual(order.direction, -1)
        self.assertEqual(order.current_stop_price, expected)

    def test_initial_percentage_sell(self):
        order = TrailingStopOrder(size=2, trail_value=-20, initial_price=100, trail_type='percentage')
        expected = 100 - (-0.2 * 100)
        self.assertAlmostEqual(order.current_stop_price, expected)

    def test_update_and_trigger_absolute_buy(self):
        order = TrailingStopOrder(size=2, trail_value=10, initial_price=100, trail_type='absolute')
        order.update_stop_price(110)
        self.assertEqual(order.current_stop_price, 100)
        order.update_stop_price(105)
        self.assertEqual(order.current_stop_price, 100)
        triggered = order.check_order_trigger(95)
        self.assertTrue(triggered)
        self.assertEqual(order.close_price, 95)
        self.assertEqual(order.gain, 2 * (95 - 100))

    def test_update_and_trigger_absolute_sell(self):
        order = TrailingStopOrder(size=4, trail_value=-10, initial_price=100, trail_type='absolute')
        order.update_stop_price(90)
        self.assertEqual(order.current_stop_price, 100)
        order.update_stop_price(95)
        self.assertEqual(order.current_stop_price, 100)
        triggered = order.check_order_trigger(150)
        self.assertTrue(triggered)
        self.assertEqual(order.close_price, 150)
        self.assertEqual(order.gain, 4 * (-50))

    def test_update_and_trigger_percentage_sell(self):
        order = TrailingStopOrder(size=2, trail_value=-20, initial_price=100, trail_type='percentage')
        order.update_stop_price(80)
        self.assertAlmostEqual(order.current_stop_price, 96)
        triggered = order.check_order_trigger(96)
        self.assertTrue(triggered)
        self.assertEqual(order.close_price, 96)
        self.assertEqual(order.gain, 2 * 4)

    def test_zero_trail_value(self):
        order = TrailingStopOrder(size=1, trail_value=0, initial_price=50, trail_type='absolute')
        self.assertEqual(order.direction, 0)
        self.assertEqual(order.current_stop_price, 50)
        triggered = order.check_order_trigger(50)
        self.assertTrue(triggered)
        self.assertEqual(order.gain, 0)

    def test_invalid_trail_type(self):
        with self.assertRaises(ValueError):
            TrailingStopOrder(size=1, trail_value=10, initial_price=100, trail_type='invalid')


if __name__ == '__main__':
    unittest.main()
