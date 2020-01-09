# tweet_sentiment

### Installation

```bash
pip install --user --upgrade git+git://github.com/dmbeskow/tweet_sentiment.git
```

### Sentiment Prediction

Below is the basic usage

```python
import tweet_sentiment
pred = tweet_sentiment.sentiment_prob(list_of_strings)
```

The tokenizer will limit to 120 tokens.  Any additional tokens will be eliminated. Tweets less than 120 tokens will be padded.

The returned probability will be between 0 and 1, with predictions closer to 0 being more negative and predictions closer to 1 being more positive.
