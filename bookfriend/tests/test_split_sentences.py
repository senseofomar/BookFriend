def split_sentences(text:str)->list[str]:
    """
    1. Take a text
    2. split it using split()
    3. Store it into a new variable
       """
    words = text.split()
    return words

def test_split_sentences():
    test = "Hello, How are you?"
    expected = ["Hello,", "How", "are", "you?"]
    assert split_sentences(test) == expected