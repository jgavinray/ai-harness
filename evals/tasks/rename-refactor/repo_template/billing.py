from items import line_items


def calc_total(order):
    return sum(price * qty for price, qty in line_items(order))
