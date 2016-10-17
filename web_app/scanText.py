import gzip
import string
from nltk.corpus import stopwords
import nltk

class scanText( object ):

    def __init__( self ):
        self.parser = iter( self.parse() )
        self.stop = set(stopwords.words('english'))
        self.punctuations = string.punctuation
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def parse( self ):

        g = gzip.open('reviews_Books_5.json.gz', 'r')

        for l in g:
            yield eval(l)

    def cntUppercases( self, review ):
        tokens = review.split()
        upperCases = [sum([1 for c in token if c.isupper()]) for token in tokens]
        return sum( upperCases )

    def scan( self, text ):
        tokens = text.split()

        uppercases = []
        puncs_all = []
        puncs_unique = []
        unique_words = []

        for token in tokens:
            cntU = 0

            word = token.strip(self.punctuations).lower()
            if word not in self.stop:
                unique_words.append( word )

            for c in token:
                if c.isupper():
                    cntU += 1
                if c in self.punctuations:
                    puncs_all.append(c)

                    if c not in puncs_unique:
                        puncs_unique.append(c)

            uppercases.append( cntU )

        n_words = len(tokens)
        n_unique_words = len(unique_words)
        n_upperCases = sum(uppercases)
        n_puncs = len(puncs_all)
        n_puncs_unique = len(puncs_unique)

        if '(' and ')' in puncs_all:
            isBrackets = '1'
        else:
            isBrackets = '0'

        # print puncs_all

        sentences = self.nltk_splitter.tokenize(text)
        n_sentences = len(sentences)
        if n_sentences == 0:
            print text
        avg_wordsPerSent = float(n_words) / (n_sentences+1)

        # print len(puncs_all), ' '.join(puncs_all)
        # print len(puncs_unique), ' '.join(puncs_unique)
        return n_sentences, avg_wordsPerSent, n_words, n_unique_words, n_upperCases, n_puncs, n_puncs_unique, isBrackets#, uppercases, puncs_all, puncs_unique

    def split( self, text  ):
        sentences = self.nltk_splitter.tokenize(text)
        print sentences
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        n_sentences = len( sentences )
        print tokenized_sentences
        print n_sentences

        return tokenized_sentences


if __name__ == "__main__":
    scanT = scanText()

    s1 = 'I was new to Python and fairly new to programming when I read this book. This book is extremely clear and well-written. It introduces a novice to the foundational concepts of computer science. There are many great examples and activities that the reader can jump into almost immediately. I had already written my first Python program after 10 minutes of reading.'
    s2 = "I'm rather new to CS, am a major in it as well. I dabble in various languages, which range from: Python to PHP to MooTools. This book teaches you theoretical computer science concepts as well as the art of programming via Python 3. It doesn't have a ton of pages and it executes its goal very well, which is to introduce students to the field of computer science and to focus on problem solving instead of complicated syntax and compiler issues. The programs are very mathematically based and it's a bit tedious (I find most technical subjects are), but the examples are fascinating if you love the idea of what can be computed. You'll broaden your mind and be ready to dive into actual programming after doing the exercises in this book."
    s3 = "The text is easy to read, the examples are well done, and it has proved to be a VERY useful book in the classroom. But wait, there's more. I emailed Dr. Zelle (the author) with a couple of questions and he responded quickly and was quite helpful. What more could you possibly ask for? An affordable text book, you say? Look no further! At this price and with these features, you simply CAN'T GO WRONG!!!"
    s4 = "I had a very tiny idea of what probability is before getting the book. The book is *shines* in giving intuitive understanding for the principals, ideas and theorems. The pictures provided are *really* good to make most of the concepts clear and intuitive."
    s5 = "I also tried with Schaum's (Outline) of Probability but it's much much worse than this one. DON'T PICK that book!!! NEVER!!!"

    print s1
    print ""
    print ""
    print "n_sentences, avg_wordsPerSent, n_words, n_unique_words, n_upperCases, n_puncs, n_puncs_unique, isBrackets:{}".format(scanT.scan(s1))
    print ""

    # list_sentences = scanW.split(s1)
    # print list_sentences
    # print len(list_sentences)

    # s3 = "I'm rather new to CS, am a major in it as well. I dabble in various languages, which range from: Python to PHP to MooTools. This book teaches you theoretical computer science concepts as well as the art of programming via Python 3. It doesn't have a ton of pages and it executes its goal very well, which is to introduce students to the field of computer science and to focus on problem solving instead of complicated syntax and compiler issues. The programs are very mathematically based and it's a bit tedious (I find most technical subjects are), but the examples are fascinating if you love the idea of what can be computed. You'll broaden your mind and be ready to dive into actual programming after doing the exercises in this book."
    # # print cuc.count(s3)
    #
    # s4 = 'While the "IDLE" compiler used by this book is rather devoid of features, it is lightweight and easy to use, which is always a plus. Students will likely switch to a more feature-rich compiler as they progress with Python.'
    # # print cuc.count(s4)
    #
    # N = 12
    # for _ in xrange( N ):
    #     review = cuc.parser.next()
    #     s5 = review['reviewText']
    #     print s5
    #     print cuc.count(s5)
    #     raw_input("Press Enter to continue...")
