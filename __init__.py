from .nodes import BuildLocalRunKey, CommitLocalRunImages


NODE_CLASS_MAPPINGS = {
    "LocalRunReceiptKey": BuildLocalRunKey,
    "LocalRunReceiptCommitImage": CommitLocalRunImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalRunReceiptKey": "Local Run Receipts: Build Run Key",
    "LocalRunReceiptCommitImage": "Local Run Receipts: Commit Image Run",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
