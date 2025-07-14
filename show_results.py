import pandas as pd

# Read the results
df = pd.read_csv('siamese_training_results.csv')

print("="*20, "Siamese Training Results", "="*20)
print()
print(f"| {'Loss Type':^12} | {'Accuracy':^9} | {'Precision':^9} | {'Recall':^8} | {'F1 Score':^9} |")
print("|" + "-"*13 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*11 + "|")
for _, row in df.iterrows():
    print(f"| {row['loss_type']:^12} | {row['accuracy']:^9.4f} | {row['precision']:^9.4f} | {row['recall']:^8.4f} | {row['f1_score']:^9.4f} |")
print()
print("="*60)