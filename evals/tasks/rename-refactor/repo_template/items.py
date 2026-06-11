def line_items(order):
    return [(item["price"], item["qty"]) for item in order["items"]]
