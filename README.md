# A3

## Dependencies

```bash
pip install -r requirements.txt
```

## Implementation

Using NLTK to apply **Extractive Summarization** with two approaches:

1. **Weighted sentence**: calculate weight for each sentence based on the frequency of each word in the whole text.
2. **Text rank**: inspired by *Page Rank*, similarity between any two sentences is used as an equivalent to the web page transition probability.

The detail of implementation is in the `summarization.py`.

## Web app

Using `streamlit` package for web app. After install all dependencies, use the following command to run:

```bash
streamlit run app.py
```
