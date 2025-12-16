import torch
import sys
sys.path.append('/Users/danieltoberman/Documents/git/Thesis/SSL')
from SingleTinyIPDnet import SingleTinyIPDnet

# Test with input_size=18
model = SingleTinyIPDnet(input_size=18, hidden_size=128)

# Create dummy input: (batch=2, channels=18, freq=256, time=200)
x = torch.randn(2, 18, 256, 200)

print(f"Input shape: {x.shape}")
output = model(x)
print(f"Output shape: {output.shape}")
print(f"Output dimensions: (batch={output.shape[0]}, time={output.shape[1]}, sources={output.shape[2]}, freq={output.shape[3]}, features={output.shape[4]})")
