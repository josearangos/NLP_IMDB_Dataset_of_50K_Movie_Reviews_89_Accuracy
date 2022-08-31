from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

class Feature_extractor:
    """
    This class vectorizes the data


    Args:
            df_data_cleaned (PandasDataFrame): Pandas dataFrame with clean data
            column_text_name (str): Name of the column that need to be preprocessed
            column_label (str): Name of the column that has the classes
            test_size (float, optional): Size of the test dataset. Defaults to 0.1.

    Methods: 
    create_countVectorizer(self, path_save_model:str,max_df=1,binary=False,ngram_range=(1,3))
	    This method create a count vectorizer representation

    create_tfidf(self,path_save_model:str,max_df=1,ngram_range=(1,3))
        This method create a count TFIDF representation


    load_feature_model(self,path_model:str)
        This method load the CountVectorizer o TfidfVectorizer model, ready to transform data 
    """
    

    def __init__(self,df_data_cleaned, column_text_name:str,column_label:str,test_size=0.1) -> None:
        """This class vectorizes the data

        Args:
            f_data_cleaned (PandasDataFrame): Pandas dataFrame with clean
            column_text_name (str): Name of the column that need to be preprocessed
            column_label (str): Name of the column that has the classes
            test_size (float, optional): Size of the test dataset. Defaults to 0.1.
        """

        train_dataset, test_dataset, train_data_label, test_data_label = train_test_split(df_data_cleaned[column_text_name], df_data_cleaned[column_label], test_size=test_size, random_state=42)



        #train_dataset, validation_dataset,train_data_label, validation_data_label  = train_test_split(train_dataset, train_data_label, test_size=test_size, random_state=42)

        self.train_dataset = train_dataset
        self.train_data_label = train_data_label
        
        #self.validation_dataset = validation_dataset
        #self.validation_data_label = validation_data_label

        self.test_dataset = test_dataset
        self.test_data_label = test_data_label
        
        


    def create_countVectorizer(self, path_save_model:str,max_df=1.0,binary=False,ngram_range=(1,1)):
        """
        This method create a count vectorizer representation

        Args:
            path_save_model (str): Path where it will be save the countvectorizer model in pickle format
            max_df (int, optional): When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold . Defaults to 1.
            binary (bool, optional): If True, all non zero counts are set to 1.. Defaults to False.
            ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.. Defaults to (1,3).
        """

        self.count_vector = CountVectorizer(max_df=max_df,binary=binary,ngram_range=ngram_range)
        #1.Representation CountVectorizer
        #transformed train texts
        if (binary):
            #Train dataset is used to fit de countVectorize
            self.count_vector_train_bn = self.count_vector.fit_transform(self.train_dataset)
            
            #transformed test texts
            #validation
            #self.count_vector_validation_bn = self.count_vector.transform(self.validation_dataset)
            #Test
            self.count_vector_test_bn = self.count_vector.transform(self.test_dataset)
        else:
            self.count_vector_train = self.count_vector.fit_transform(self.train_dataset)
            #transformed test texts
            #self.count_vector_validation = self.count_vector.transform(self.validation_dataset)
            self.count_vector_test= self.count_vector.transform(self.test_dataset)

        #Save Count_Vector pickle file
        pickle.dump(self.count_vector, open(path_save_model, "wb"))


    
    def create_tfidf(self,path_save_model:str,max_df=1.0,ngram_range=(1,1)):
        """
        This method create a count TFIDF representation
        
        33087    0.9 - 1.0
        33087    0.8
        33087    0.7
        33086    0.6
        33084    0.5
        33082    0.4 
        Args:
            path_save_model (str): Path where it will be save the TFIDF model in pickle format
            max_df (int, optional): When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold . Defaults to 1.
            ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.. Defaults to (1,3).
        """

        self.tfidf_vector = TfidfVectorizer(max_df=max_df,use_idf=True,ngram_range=ngram_range)
        #transformed train texts
        self.tfidf_vector_train = self.tfidf_vector.fit_transform(self.train_dataset)
        
        #transformed validation texts
        #self.tfidf_vector_validation = self.tfidf_vector.transform(self.validation_dataset)

        #transformed test texts
        self.tfidf_vector_test = self.tfidf_vector.transform(self.test_dataset)

        #Save tfidf_vector pickle file
        pickle.dump(self.tfidf_vector_test, open(path_save_model, "wb"))
    

    def load_feature_model(self,path_model:str):
        """
        This method load the CountVectorizer o TfidfVectorizer model, ready to transform data 

        Args:
            path_model (str): Path where is locate the model

        Returns:
            TfidfVectorizer/CountVectorizer: The model CountVectorizer o TfidfVectorizer
        """
        return pickle.load(open(path_model, "rb"))