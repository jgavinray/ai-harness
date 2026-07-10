"""In-memory stock reservations."""


class Inventory:
    def __init__(self, stock):
        self.stock = dict(stock)
        self.reserved = {}

    def reserve(self, sku, quantity):
        available = self.stock.get(sku, 0) - self.reserved.get(sku, 0)
        if quantity > available:
            return False
        self.reserved[sku] = self.reserved.get(sku, 0) + quantity
        return True

    def release(self, sku, quantity):
        held = self.reserved.get(sku, 0)
        self.reserved[sku] = max(0, held - quantity)
