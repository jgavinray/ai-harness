"""Order billing: totals, discounts, and tax."""

TAX_RATE = 0.08


def line_total(price, quantity):
    return price * quantity


def apply_discount(subtotal, discount):
    if discount < 0:
        raise ValueError("discount must be non-negative")
    return subtotal - discount


def order_total(items, discount):
    subtotal = sum(line_total(p, q) for p, q in items)
    discounted = apply_discount(subtotal, discount)
    return round(discounted * (1 + TAX_RATE), 2)
