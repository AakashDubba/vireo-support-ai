import pandas as pd
detail = pd.read_csv('data/human_labels_detail.csv')
wrong = detail[detail['human_label'] == 'Product Enquiry & General']
print('== Messages labelled as Product Enquiry & General ==')
for _, r in wrong.head(20).iterrows():
    print(f"ticket={r['ticket_id']} | tag={r['existing_tag']}")
    print(f"  msg: {r['customer_message'][:140]}")
    print()

print('\n== All labels ==')
print(detail['human_label'].value_counts().to_string())
