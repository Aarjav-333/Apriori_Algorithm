import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = [
    ['Eggs', 'Cola', 'Juice'],
    ['Eggs', 'Chips', 'Juice'],
    ['Eggs', 'Chips', 'Cola'],
    ['Eggs', 'Chips', 'Juice', 'Cheese'],
    ['Cola', 'Chips'],
    ['Eggs', 'Cola', 'Cheese'],
    ['Cola', 'Eggs', 'Cheese'],
    ['Cola', 'Eggs', 'Chips', 'Cheese'],
    ['Cola', 'Eggs', 'Chips', 'Juice'],
    ['Eggs', 'Cola', 'Chips'],
    ['Chips', 'Juice', 'Cola'],
    ['Eggs', 'Chips', 'Juice'],
    ['Eggs', 'Cola', 'Cheese'],
    ['Eggs', 'Chips', 'Juice'],
    ['Eggs', 'Chips', 'Juice'],
    ['Eggs', 'Cola', 'Chips', 'Juice'],
    ['Eggs', 'Chips', 'Cheese'],
    ['Eggs', 'Cola', 'Chips', 'Juice'],
    ['Eggs', 'Chips', 'Juice'],
    ['Eggs', 'Cola', 'Chips', 'Juice'],
    ['Eggs', 'Chips', 'Juice'],
    ['Eggs', 'Cola', 'Chips', 'Juice'],
    ['Eggs', 'Chips', 'Juice'],
    ['Eggs', 'Cola', 'Chips', 'Juice'],
    ['Eggs', 'Chips', 'Cola'],
    ['Eggs', 'Cola', 'Cheese', 'Juice']
]

df = pd.DataFrame(data)

transactions = df.stack().groupby(level=0).apply(list)

unique_items = sorted(set(item for sublist in transactions for item in sublist))

encoded_data = []
for transaction in transactions:
    encoded_data.append([1 if item in transaction else 0 for item in unique_items])

encoded_df = pd.DataFrame(encoded_data, columns=unique_items).astype(bool)

frequent_itemsets = apriori(encoded_df, min_support=0.3, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

rules["lift"] = rules["confidence"] / (rules["consequent support"])

frequent_itemsets_selected = frequent_itemsets[['support', 'itemsets']]
association_rules_selected = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

print("Selected Frequent Itemsets:")
print(frequent_itemsets_selected)
print("\nSelected Association Rules:")
print(association_rules_selected)

frequent_itemsets_selected.to_csv('frequent_itemsets_selected.csv', index=False)
association_rules_selected.to_csv('association_rules_selected.csv', index=False)
