from billing import order_total
from inventory import Inventory
from report import summarize


def place_order(order_id, items, discount, inventory):
    for sku, (_, quantity) in items.items():
        if not inventory.reserve(sku, quantity):
            for held, (_, quantity) in items.items():
                if held == sku:
                    break
                inventory.release(held, quantity)
            return None
    pairs = list(items.values())
    total = order_total(pairs, discount)
    return summarize(order_id, pairs, total)


if __name__ == "__main__":
    inv = Inventory({"apple": 10, "pear": 5})
    print(place_order("A-1", {"apple": (1.25, 4), "pear": (2.10, 2)}, 1.00, inv))
