from setuptools import setup

setup(name='tweet_sentiment',
      version='0.1',
      description='This package conducts sentiment analysis (sentiment polarity).  The model is an LSTM Deep Learning Model trained on the Kaggle Sentiment140 Data',
      url='http://github.com/dbeskow/tweet_sentiment',
      author='David Beskow',
      author_email='dnbeskow@gmail.com',
      license='MIT',
      packages=['tweet_sentiment'],
      install_requires=[
              'tensorflow',
              'keras',
              'progressbar2',
              'matplotlib',
              'pandas'
              ],
      # scripts=['bin/stream_content', 'bin/stream_geo'],
      include_package_data = True,
      zip_safe=False)
