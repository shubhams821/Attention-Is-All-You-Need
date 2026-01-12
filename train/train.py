from tokenizer.tokenizer import vocabulary
from model.model import AttIsAllYouNeed

class TranslateDataset(Dataset):
    def __init__(self, df, vocabulary):
        self.df = df
        self.eng = df.iloc[:, 0]   # or df["english"]
        self.hin = df.iloc[:, 1]   # or df["hindi"]
        self.vocab = vocabulary

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        eng_tok = self.vocab.tokenize(self.eng.iloc[idx])
        hin_tok = self.vocab.tokenize(self.hin.iloc[idx])
        return {
            "eng": torch.tensor(eng_tok + [self.vocab.tokenize("EOD") for i in range(self.vocab.max_len - len(eng_tok)+1)]),
            "hin": torch.tensor([self.vocab.tokenize("SOT")] + hin_tok + [self.vocab.tokenize("EOD") for i in range(self.vocab.max_len - len(hin_tok))])
        }
train_data = TranslateDataset(train_df, vocablary)
test_data = TranslateDataset(test_df, vocablary)
valid_data = TranslateDataset(valid_df, vocablary)

train_loader = DataLoader(train_data, batch_size = 256, shuffle= True)
test_loader = DataLoader(test_data, batch_size = 256)
valid_loader = DataLoader(valid_data, batch_size = 256)

vocab_size=len(vocablary.total_char)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AttIsAllYouNeed(vocab_size, dim = 1024, depth = 4, max_len=27).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr= 1e-4)


from tqdm import tqdm
for epoch in range(5):
    model.train()
    total_loss = 0
    test_loss = 0
    for train_load in tqdm(train_loader):
        eng, hin = train_load["eng"].to(device), train_load["hin"].to(device)
        logits = model(eng, hin)
        logits = logits.reshape(-1, logits.size(-1))  # (256*4, 1000)
        targets = hin.reshape(-1)  
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for test_load in tqdm(test_loader):
        eng, hin = test_load["eng"].to(device), test_load["hin"].to(device)
        logits = model(eng, hin)
        logits = logits.reshape(-1, logits.size(-1))  # (256*4, 1000)
        targets = hin.reshape(-1)  
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)
        test_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f} Test Loss: {test_loss/len(test_loader):.4f}")
