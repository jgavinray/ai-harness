"""In-memory stock reservations."""


class Inventory:
    def __init__(self, stock):
        self.stock = dict(stock)
        self.reserved = {}

    def available(self, sku):
        return self.stock.get(sku, 0) - self.reserved.get(sku, 0)

    def reserve(self, sku, quantity):
        if quantity > self.available(sku):
            return False
        self.reserved[sku] = self.reserved.get(sku, 0) + quantity
        return True

    def release(self, sku, quantity):
        held = self.reserved.get(sku, 0)
        self.reserved[sku] = max(0, held - quantity)
