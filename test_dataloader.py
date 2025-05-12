from util.tokenizer import Tokenizer
from util.data_loader import HuggingFaceDataLoader as DataLoader
from conf import *

def main():
    # Initialize tokenizer and loader
    tokenizer = Tokenizer()
    loader = DataLoader(
        ext=('.en', '.de'),
        tokenize_en=tokenizer.tokenize_en,
        tokenize_de=tokenizer.tokenize_de,
        init_token='<sos>',
        eos_token='<eos>'
    )

    # Load and process dataset
    train_data, valid_data, test_data = loader.make_dataset()
    loader.build_vocab(train_data=train_data, min_freq=2)
    train_iter, valid_iter, test_iter = loader.make_iter(train_data, valid_data, test_data, batch_size, device)

    # Get a single batch
    print("\nFetching a batch from the training iterator...")
    for src_batch, trg_batch in train_iter:
        print("Source batch shape:", src_batch.shape)
        print("Target batch shape:", trg_batch.shape)
        print("\nSample source (token IDs):\n", src_batch[0])
        print("\nSample target (token IDs):\n", trg_batch[0])
        break

    # Print vocab sizes
    print(f"\nSource vocab size: {len(loader.source_vocab)}")
    print(f"Target vocab size: {len(loader.target_vocab)}")
    print(f"Source <pad> index: {loader.source_vocab['<pad>']}")
    print(f"Target <sos> index: {loader.target_vocab['<sos>']}")

if __name__ == '__main__':
    main()
