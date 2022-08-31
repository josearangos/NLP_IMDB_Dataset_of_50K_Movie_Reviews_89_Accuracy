from telnetlib import SE
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')


class DataSet:
    """
    This class cleans the data
    
    Args:
            df (PandasDataFrame): PandasDataFrame with the raw data
            column_text (str): Name of the column that need to be preprocessed
            column_label (str): Name of the column that has the classes
            class_labels (list): List of the labels, e.g: ["Positive","Negative"]
            class_numbers (list): List of the class numbers of labels, e.g [1,0]

    Methods:   

    create_sample_dataset(self,n=5000)
        This method creates a new DataFrame, it randomly selects n samples by class
    
    remove_html(self,text:str)
	This method delete html tags from a string

    remove_url(self, text:str)
        This method delete urls from a string

    remove_special_characters(self, text:str)
        This method delete special characters from a string such as, e.g: ,.*,!

    remove_emojics(self, text:str)
        This method delete emojics from a string

    remove_blanck_spaces(self,text:str)
        This method delete blank spaces from a string

    replace_abbreviations(self, text:str)
        This method replace de abbrevations from a string, e.g: don't => do not


    remove_stopWords(self,text:str)
        This method remove the stop words, such as: e.g, i,in,the,a,etc.


    lemmatization(self, reviews:str)
        This method applies the lemmatization process to a word

    stemming(self, reviews:str)
        This method applies the Stemming process to a word

    clean_data(self)
        This method cleans the data applying the previously mentioned methods

    label_encode(self)
        This method replace string class by a number, e.g: Positive => 1, Negative=>0

    data_preprocess(self,PATH_SAVE:str,stemming=True,stop_words=True)
        This method delete stop-words and apply Stemming/Lemmatization process

    """

    # NLP Objetcs to preprocessing
    stop_words = stopwords.words('english')
    stemmer = PorterStemmer()
    lemma = WordNetLemmatizer()

    # Regex 
    html_regex = re.compile(r'<.*?>')
    url_regex = re.compile(r"http://\S+|www\.\S+")
    special_characters_regex = re.compile(r'[^a-zA-Z ]')
    emojic_regex= re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    def __init__(self,df, column_text:str, column_label:str, class_labels:list, class_numbers:list) -> None:
        """_summary_

        Args:
            df (PandasDataFrame): PandasDataFrame with the raw data
            column_text (str): Name of the column that need to be preprocessed
            column_label (str): Name of the column that has the classes
            class_labels (list): List of the labels, e.g: ["Positive","Negative"]
            class_numbers (list): List of the class numbers of labels, e.g [1,0]
        """
        
        self.df = df
        self.column_text=column_text
        self.column_label = column_label

        self.class_labels = class_labels
        self.class_numbers = class_numbers

        print("Creating sample dataset....")
        self.create_sample_dataset()
        print("Sample dataset created")

    
    def create_sample_dataset(self,n=5000):
        """
        This method creates a new DataFrame, it randomly selects n samples by class

        Args:
            n (int, optional): Number of samples by class . Defaults to 5000.

        Returns:
            PandasDataFrame: DataFrame with number of samples equal to n*2, Defaults shape 10.000 x 2 
        """

        df_positive = self.df[self.df.sentiment == self.class_labels[0]]
        df_negative = self.df[self.df.sentiment == self.class_labels[1]]
        df_positive = df_positive.sample(n)
        df_negative = df_negative.sample(n)
        df_dataset = pd.concat([df_positive, df_negative]).sample(frac=1).reset_index(drop=True)
        self.df_dataset = df_dataset

        return self.df_dataset

    # NLP Cleaning 

    def remove_html(self,text:str) -> str:
        """
        This method delete html tags from a string
        Args:
            text (str): text to edit

        Returns:
            str: new text edited
        """
        return self.html_regex.sub(r' ',text)
    
    def remove_url(self, text:str) -> str:
        """
        This method delete urls from a string

        Args:
            text (str): text to edit

        Returns:
            str: new text edited
        """

        return self.url_regex.sub(r' ',text)

    def remove_special_characters(self, text:str)-> str:
        """
        This method delete special characters from a string such as, e.g: ,.*,!

        Args:
            text (str): text to edit

        Returns:
            str: new text edited
        """
        return self.special_characters_regex.sub(r' ',text)

    def remove_emojics(self, text:str) -> str:
        """
        This method delete emojics from a string


        Args:
            text (str): text to edit

        Returns:
            str: new text edited
        """


        return self.emojic_regex.sub(r' ',text)

    def remove_blanck_spaces(self,text:str) -> str:
        """
        This method delete blank spaces from a string

        Args:
            text (str): text to edit

        Returns:
            str: new text edited   
                                    
        """
        new_text = text.split(" ")

        while "" in new_text:
            if "" in new_text:
                new_text.remove("")
        new_text = " ".join(new_text)
        return new_text
    
    def replace_abbreviations(self, text:str) -> str:
        """
        This method replace de abbrevations from a string, e.g: don't => do not

        Args:
            text (str): text to edit

        Returns:
            str: new text edited
        """


        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"there's", "there is", text)
        text = re.sub(r"We're", "We are", text)
        text = re.sub(r"That's", "That is", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"they're", "they are", text)
        text = re.sub(r"Can't", "Cannot", text)
        text = re.sub(r"wasn't", "was not", text)
        text = re.sub(r"don\x89Ûªt", "do not", text)
        text= re.sub(r"aren't", "are not", text)
        text = re.sub(r"isn't", "is not", text)
        text = re.sub(r"What's", "What is", text)
        text = re.sub(r"haven't", "have not", text)
        text = re.sub(r"hasn't", "has not", text)
        text = re.sub(r"There's", "There is", text)
        text = re.sub(r"He's", "He is", text)
        text = re.sub(r"It's", "It is", text)
        text = re.sub(r"You're", "You are", text)
        text = re.sub(r"I'M", "I am", text)
        text = re.sub(r"shouldn't", "should not", text)
        text = re.sub(r"wouldn't", "would not", text)
        text = re.sub(r"i'm", "I am", text)
        text = re.sub(r"I\x89Ûªm", "I am", text)
        text = re.sub(r"I'm", "I am", text)
        text = re.sub(r"Isn't", "is not", text)
        text = re.sub(r"Here's", "Here is", text)
        text = re.sub(r"you've", "you have", text)
        text = re.sub(r"you\x89Ûªve", "you have", text)
        text = re.sub(r"we're", "we are", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"couldn't", "could not", text)
        text = re.sub(r"we've", "we have", text)
        text = re.sub(r"it\x89Ûªs", "it is", text)
        text = re.sub(r"doesn\x89Ûªt", "does not", text)
        text = re.sub(r"It\x89Ûªs", "It is", text)
        text = re.sub(r"Here\x89Ûªs", "Here is", text)
        text = re.sub(r"who's", "who is", text)
        text = re.sub(r"I\x89Ûªve", "I have", text)
        text = re.sub(r"y'all", "you all", text)
        text = re.sub(r"can\x89Ûªt", "cannot", text)
        text = re.sub(r"would've", "would have", text)
        text = re.sub(r"it'll", "it will", text)
        text = re.sub(r"we'll", "we will", text)
        text = re.sub(r"wouldn\x89Ûªt", "would not", text)
        text = re.sub(r"We've", "We have", text)
        text = re.sub(r"he'll", "he will", text)
        text = re.sub(r"Y'all", "You all", text)
        text = re.sub(r"Weren't", "Were not", text)
        text = re.sub(r"Didn't", "Did not", text)
        text = re.sub(r"they'll", "they will", text)
        text = re.sub(r"they'd", "they would", text)
        text = re.sub(r"DON'T", "DO NOT", text)
        text = re.sub(r"That\x89Ûªs", "That is", text)
        text = re.sub(r"they've", "they have", text)
        text = re.sub(r"i'd", "I would", text)
        text = re.sub(r"should've", "should have", text)
        text = re.sub(r"You\x89Ûªre", "You are", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"Don\x89Ûªt", "Do not", text)
        text = re.sub(r"we'd", "we would", text)
        text = re.sub(r"i'll", "I will", text)
        text = re.sub(r"weren't", "were not", text)
        text = re.sub(r"They're", "They are", text)
        text = re.sub(r"Can\x89Ûªt", "Cannot", text)
        text = re.sub(r"you\x89Ûªll", "you will", text)
        text = re.sub(r"I\x89Ûªd", "I would", text)
        text = re.sub(r"let's", "let us", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"don't", "do not", text)
        text = re.sub(r"you're", "you are", text)
        text = re.sub(r"i've", "I have", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"i'll", "I will", text)
        text = re.sub(r"doesn't", "does not",text)
        text = re.sub(r"i'd", "I would", text)
        text = re.sub(r"didn't", "did not", text)
        text = re.sub(r"ain't", "am not", text)
        text = re.sub(r"you'll", "you will", text)
        text = re.sub(r"I've", "I have", text)
        text = re.sub(r"Don't", "do not", text)
        text = re.sub(r"I'll", "I will", text)
        text = re.sub(r"I'd", "I would", text)
        text = re.sub(r"Let's", "Let us", text)
        text = re.sub(r"you'd", "You would", text)
        text = re.sub(r"It's", "It is", text)
        text = re.sub(r"Ain't", "am not", text)
        text = re.sub(r"Haven't", "Have not", text)
        text = re.sub(r"Could've", "Could have", text)
        text = re.sub(r"youve", "you have", text)  
        text = re.sub(r"donå«t", "do not", text) 

        return text


    # NLP Preprocessing
    def remove_stopWords(self,text:str) -> str:
        """
        This method remove the stop words, such as: e.g, i,in,the,a,etc.

        Args:
            text (str): text to edit

        Returns:
            str: new text edited
        """
        new_text = []
        
        for txt in text.split(" "):
            if txt not in self.stop_words and len(txt)>1:
                new_text.append(txt)

        new_text = " ".join(new_text)       

        return new_text

    def lemmatization(self, reviews:str) -> str:
        """
        This method applies the lemmatization process to a word

        Args:
            reviews (str): text to edit

        Returns:
            str: new text edited
        """
        texts_lemm = []
        for review in  reviews.split(" "):
            texts_lemm.append(self.lemma.lemmatize(review))

        texts_lemm = " ".join(texts_lemm)

        return texts_lemm
   
    def stemming(self, reviews:str) -> str:
        """
        This method applies the Stemming process to a word

        Args:
            reviews (str): text to edit

        Returns:
            str: new text edited
        """

        texts_stem = []
        for review in reviews.split(" "):
            texts_stem.append(self.stemmer.stem(review))

        texts_stem = " ".join(texts_stem)
        return texts_stem



    

    def clean_data(self):
        """
        This method cleans the data applying the previously mentioned methods

        Returns:
            PandasDataFrame: New DataFrame with the clean data, ready to apply NLP process
        """
        df_cleaned = self.df_dataset.copy()
        df_cleaned[self.column_text] = df_cleaned[self.column_text].apply(lambda text: self.replace_abbreviations(text))
        df_cleaned[self.column_text] = df_cleaned[self.column_text].apply(lambda text: self.remove_url(text)) 
        df_cleaned[self.column_text] = df_cleaned[self.column_text].apply(lambda text: self.remove_html(text))        
        df_cleaned[self.column_text] = df_cleaned[self.column_text].apply(lambda text: self.remove_emojics(text))         
        df_cleaned[self.column_text] = df_cleaned[self.column_text].apply(lambda text: self.remove_special_characters(text))
        df_cleaned[self.column_text] = df_cleaned[self.column_text].apply(lambda text:self.remove_blanck_spaces(text))
        df_cleaned[self.column_text] = df_cleaned[self.column_text].str.lower()
        
        self.df_cleaned = df_cleaned
        return df_cleaned

    def label_encode(self):
        """
        This method replace string class by a number, e.g: Positive => 1, Negative=>0
        """

        for c in range(len(self.class_labels)):
            self.df_cleaned[self.column_label] =  self.df_cleaned[self.column_label].replace(self.class_labels[c],self.class_numbers[c])


    def data_preprocess(self,PATH_SAVE:str,stemming=True,lemma = False, stop_words=True):
        """
        This method delete stop-words and apply Stemming/Lemmatization process

        Args:
            PATH_SAVE (str): PATH to save de dataFrame after to clean and preprocessing
            stemming (bool, optional): If the data need to Stemming process. Defaults to True.
            stop_words (bool, optional): If the data need to delete the stop words. Defaults to True.

        Returns:
            PandasDataFrame: New DataFrame with all data cleaned and preprocessed 
        """
        
        print("1.Cleaning dataset....")
        self.clean_data()
        print("2.Encoding label......")
        self.label_encode()
        
        
        if(stop_words):
            print("3.Remove stopwords.....") 
            self.df_cleaned[self.column_text] = self.df_cleaned[self.column_text].apply(lambda x:self.remove_stopWords(x)) 
                
        if(stemming):
            print("4.Stemming .....")
            self.df_cleaned[self.column_text] = self.df_cleaned[self.column_text].apply(lambda x:self.stemming(x))

        if(lemma):
            print("4.Lemmatization....")
            self.df_cleaned[self.column_text] = self.df_cleaned[self.column_text].apply(lambda x:self.lemmatization(x))
           
        if PATH_SAVE:
            print("Save file....in",PATH_SAVE)
            self.df_cleaned.to_csv(PATH_SAVE,index=False)

        return self.df_cleaned