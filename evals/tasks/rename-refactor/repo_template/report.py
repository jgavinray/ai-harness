from billing import calc_total


def summary(order):
    return f"order {order['id']}: total={calc_total(order)}"


if __name__ == "__main__":
    order = {"id": 7, "items": [{"price": 5, "qty": 2}, {"price": 1, "qty": 3}]}
    print(summary(order))
    assert calc_total(order) == 13
