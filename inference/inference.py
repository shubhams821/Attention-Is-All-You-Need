def inference(model, vocab, eng):
    eng_tok = torch.tensor(vocab.tokenize(eng)).unsqueeze(0)
    idx = torch.tensor([vocab.tokenize("SOT")]).unsqueeze(0)
    print(idx.shape, eng_tok.shape)
    for i in range(10):
        print(i)
        logits = model(eng_tok.to(device), idx.to(device))

        # logits: (B, T, V) → take last time step
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        # print(next_token, idx, next_token.shape, idx.shape)

        # concat along sequence dimension
        idx = torch.cat([idx.to(device), next_token.to(device)], dim=1)
    idx = idx.squeeze(0).tolist()
    return vocab.detokenize(idx)


translated_word = inference(model, vocablary, "bindhya")
print(translated_word)
