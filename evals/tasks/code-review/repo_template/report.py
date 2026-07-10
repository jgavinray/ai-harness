"""Plain-text order summaries."""


def summarize(order_id, items, total):
    lines = [f"Order {order_id}"]
    for price, quantity in items:
        lines.append(f"  {quantity} x {price:.2f}")
    lines.append(f"Total: {total:.2f}")
    return "\n".join(lines)
