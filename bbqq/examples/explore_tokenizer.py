from transformers import BertTokenizer

def main():
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    sent = "뉴스기사"
    print(tokenizer.tokenize(sent))
    print(tokenizer.decode(2))
    print(tokenizer.decode(0))
    print(tokenizer.decode(3))



if __name__ == '__main__':
    main()