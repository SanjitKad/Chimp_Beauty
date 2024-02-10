from src.testing.suggest_products import SuggestProducts


def main(username):
    suggested = SuggestProducts()
    suggested_products = suggested.suggest_products(username)
    print(suggested_products)
    return suggested_products