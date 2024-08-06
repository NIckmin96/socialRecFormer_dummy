import torch

# Example boolean tensor
bool_tensor = torch.tensor([True, False, True, False, True, True, False, False, True, False])


# Example item embedding tensor
item_embedding_tensor = torch.rand((10, 64))

# Select only the rows where bool_tensor is True
selected_item_embeddings = item_embedding_tensor[bool_tensor]
selected_item_embeddings2 = item_embedding_tensor[~bool_tensor]
# Display the shapes
print("Original Item Embedding Shape:", selected_item_embeddings2.shape)
print("Selected Item Embedding Shape:", selected_item_embeddings.shape)
