import numpy as np


class TrailingStopOrder:
    def __init__(self,
                 size: float,
                 trail_value: float,
                 initial_price: float,
                 trail_type='absolute',
                 ):

        self.size_value = size
        self.trail_value = trail_value
        self.direction = np.sign(trail_value)  # 'buy' or 'sell'

        self.trail_type = trail_type  # 'absolute' or 'percentage'
        self.initial_price = initial_price
        self.close_price = None
        self.current_stop_price = self.calculate_initial_stop_price(initial_price)

        self.gain = 0.

    def calculate_initial_stop_price(self, initial_price: float):
        if self.trail_type == 'percentage':
            trail_amount = (self.trail_value / 100.) * initial_price
        else:  # absolute
            trail_amount = self.trail_value

        return initial_price - trail_amount

    def update_stop_price(self, current_price: float):
        if self.trail_type == 'percentage':
            trail_amount = (self.trail_value / 100.) * current_price
        else:  # absolute
            trail_amount = self.trail_value

        if self.direction >= 0.:  # buy
            new_stop_price = current_price - trail_amount
            if new_stop_price > self.current_stop_price:
                self.current_stop_price = new_stop_price
        else:  # sell
            new_stop_price = current_price - trail_amount
            if new_stop_price < self.current_stop_price:
                self.current_stop_price = new_stop_price

    def check_order_trigger(self, current_price: float):

        diff_price = current_price - self.initial_price
        if self.direction >= 0.:    # buy
            trigger_flag = current_price <= self.current_stop_price

        else:       # sell
            trigger_flag = current_price >= self.current_stop_price
            diff_price *= -1

        if trigger_flag:
            self.close_price = current_price
            self.gain = self.size_value * diff_price

        return trigger_flag


# Example usage
if __name__ == "__main__":


    # Simulate some market price movements
    market_prices = [100., 103., 106., 112., 107., 102., 95., 90., 88., 85., 80., 95., 100., 105., 110., 115., 120.]
    trigger_price = None
    trailing_stop_order = None

    for i, price_i in enumerate(market_prices):
        if i == 0:
            # Initialize a trailing stop order
            trailing_stop_order = TrailingStopOrder(
                size= 10.,
                initial_price=100,
                trail_value=-15,
                trail_type='percentage',
            )
        else:
            if trailing_stop_order.check_order_trigger(price_i):
                trigger_price = price_i
                print(
                    f"Order triggered at market price: {price_i} (iter: {i}), with stop price: {trailing_stop_order.current_stop_price}")
                break
            else:
                trailing_stop_order.update_stop_price(price_i)
                print(f"Market price: {price_i} (iter: {i}), Current stop price: {trailing_stop_order.current_stop_price:.2f}")
