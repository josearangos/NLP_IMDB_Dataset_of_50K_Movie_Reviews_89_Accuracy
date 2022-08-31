import ast 
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Experiment:
    """



    This class runs experiments uses a dataframe with the experiments, 
    which has in each row an experiment. Each experiment is composed of a model and a grid of hyper-parameters to test.
    This class creates a GridSearchCV and executes multiple trainings, 
    Then saves the best model of the experiment in a file (pkl) and the image of the confusion matrix (png),
    as well as writes in an excel file the results obtained from the experiment (metrics).

Translated with www.DeepL.com/Translator (free version)

    Args:
            df_experiments (PandasDataFrame): DataFrame with the experiments descriptions
            PATH_METRICS (str): path of the metrics file(excel)
            df_metrics (PandasDataFrame): DataFrame with the metrics
            EXPERT_ID (int): id of the experiment that will be executed
            data_representation (str): name of data representation tfidf,CountVectorizer,ETC. 
            train_data_representation (numpy_array): train data
            train_labels (numpy_array): train label
            test_data_representation (numpy_array): test data
            test_labels (numpy_array): test label

        
    Methods: 
        metrics(self,y_true,y_pred)
            This method calculate some metrics shuch as acurracy,f1-score,precision and create confusion matrix figure.

        save_experiment(self,experiment_name:str,model_name:str,grid_search,best_parametres:dict,train_score:float,validation_score:float,test_score:float,cm_fig,number_fits:int)
            This method saves the results of an experiment:
                1. Save the confusion matrix image
                2. Save the best model
                3. Inserts in a row of an metrics excel file: the best parameters, the model, the best training, validation and test scores.
        
        run_experiment(self)
            This method executes the experiment as follows:
                1. Identify the model
                2. Prepare the hyper-parameter grid.
                2. Create the gridSearch with the model
                3. Train the gridSearch
                4. Compute metrics
                5. saves the results of the experiment

    """

    def __init__(self,df_experiments,PATH_METRICS:str,df_metrics,EXPERT_ID:int,data_representation:str,train_data_representation,train_labels,test_data_representation,test_labels):
        """
        This class runs experiments uses a dataframe with the experiments, 
        which has in each row an experiment. Each experiment is composed of a model and a grid of hyper-parameters to test.
        This class creates a GridSearchCV and executes multiple trainings, 
        Then saves the best model of the experiment in a file (pkl) and the image of the confusion matrix (png),
        as well as writes in an excel file the results obtained from the experiment (metrics).

        Args:
            df_experiments (PandasDataFrame): DataFrame with the experiments descriptions
            PATH_METRICS (str): path of the metrics file(excel)
            df_metrics (PandasDataFrame): DataFrame with the metrics
            EXPERT_ID (int): id of the experiment that will be executed
            data_representation (str): name of data representation tfidf,CountVectorizer,ETC. 
            train_data_representation (numpy_array): train data
            train_labels (numpy_array): train label
            test_data_representation (numpy_array): test data
            test_labels (numpy_array): test label
        """
        
        self.df_experiments=df_experiments
        self.PATH_METRICS = PATH_METRICS
        self.df_metrics=df_metrics
        self.EXPERT_ID = EXPERT_ID
        self.data_representation = data_representation
        self.train_data_representation = train_data_representation
        self.train_labels = train_labels
        self.test_data_representation = test_data_representation
        self.test_labels = test_labels
        self.METRIC='accuracy'
        self.FOLDS=10




    def metrics(self,y_true,y_pred):
        """
        This method calculate some metrics shuch as acurracy,f1-score,precision and create confusion matrix figure.

        Args:
            y_true (numpy_array): true classes
            y_pred (numpy_array): predict classes

        Returns:
            
            cm_fig (ConfusionMatrixDisplay: Confusion matrix figure
            accuracy (float): acurracy
            report (dict): some metrics

        """
        cm = confusion_matrix(y_true,y_pred, normalize='true')
        report = classification_report(y_true,y_pred,output_dict=True)
        cm_fig = ConfusionMatrixDisplay(confusion_matrix=cm)
        return cm_fig,report["accuracy"],report

    
    def save_experiment(self,experiment_name:str,model_name:str,grid_search,best_parametres:dict,train_score:float,validation_score:float,test_score:float,cm_fig,number_fits:int):
        """
        This method saves the results of an experiment:
            1. Save the confusion matrix image
            2. Save the best model
            3. Inserts in a row of an metrics excel file: the best parameters, the model, the best training, validation and test scores.

        Args:
            experiment_name (str): name of the experiment
            model_name (str): name of the model
            grid_search (sklearn.model_selection.GridSearchCV): Grid search model trained
            best_parametres (dict): list of the best parametres got
            train_score (float): score of the train
            validation_score (float): score of the validation
            test_score (float): score of the test
            cm_fig (ConfusionMatrixDisplay): confusion matriz figure
            number_fits (int): number of the fits done
        """

        path_confusion_matrix = "../reports/confusions_matrix/"+experiment_name+"_cm.jpg"
        path_pickel_model = "../models/"+experiment_name+".pkl"

        df_new_experiment = pd.DataFrame({
            'experiment_id':self.EXPERT_ID,
            'data_representation':self.data_representation,
            'model_name':model_name,
            'best_parametres':[best_parametres],
            'train_score':train_score,
            'validation_score':validation_score,
            'test_score':test_score, 
            'path_confusion_matrix':path_confusion_matrix,
            'path_pickel_model':path_pickel_model,
            'number_fits':number_fits
        })   

        self.df_metrics = pd.read_excel(self.PATH_METRICS)
        df_final = pd.concat([self.df_metrics,df_new_experiment])
        df_final.to_excel(self.PATH_METRICS,index = False)
        
        #save model
        joblib.dump(grid_search.best_estimator_,path_pickel_model)

        #save confusion matrix
        cm_fig.figure_.savefig(path_confusion_matrix)



    def run_experiment(self):
        """
        This method executes the experiment as follows:

        1. Identify the model
        2. Prepare the hyper-parameter grid.
        2. Create the gridSearch with the model
        3. Train the gridSearch
        4. Compute metrics
        5. saves the results of the experiment

        """              

        #load model and parametres grid
        model_name = self.df_experiments.loc[self.EXPERT_ID,"model"]
        param_grid = ast.literal_eval(self.df_experiments.loc[self.EXPERT_ID,"parametres"])
        
        number_fits = len(ParameterGrid(param_grid))*self.FOLDS + 1  
        # load model
        models = {'LogisticRegression':LogisticRegression(),'MultinomialNB':MultinomialNB(),'LinearSVC':LinearSVC(),
        'GradientBoostingClassifier':GradientBoostingClassifier(),'RandomForestClassifier':RandomForestClassifier(),
        'KNeighborsClassifier':KNeighborsClassifier(),
        'SVC':SVC(),'MLPClassifier':MLPClassifier()
        }
        model  = models[model_name]

        # Create cross validator - cross-validator
        cv = KFold(n_splits=self.FOLDS)

        # Create gridSearch
        grid_search =  GridSearchCV(model, param_grid,cv=cv,scoring = self.METRIC,refit=True,n_jobs=-1,verbose=3,return_train_score=True)

        # Fit grid
        print("#"*50)
        print("Run experiment #:",str(self.EXPERT_ID))
        print("Model =>",model_name)
        print("Parametres Grid:",param_grid)
        print("Number of fits:",number_fits)
        print("Running experiment.....")
        grid_search.fit(self.train_data_representation,self.train_labels)
        print("Experiment runned with successfull")

        best_parametres = grid_search.best_params_
        best_metric  = grid_search.best_score_

        #Train - Validation metrics
        train_score = grid_search.cv_results_["mean_train_score"][grid_search.best_index_]
        validation_score = grid_search.cv_results_["mean_test_score"][grid_search.best_index_]

        #Experiment name
        name = ""
        for p in range(len(list(best_parametres.keys()))):
            name = name + str(list(best_parametres.keys())[p])+"_"+str(list(best_parametres.values())[p]) + "_"
        name = name[:-1]

       
        
        
        
        #Predict and calculate test metrics
        test_predict = grid_search.predict(self.test_data_representation)
        cm_fig,test_score, report = self.metrics(self.test_labels,test_predict)
        cm_fig.plot(cmap=plt.cm.Blues)

        experiment_name = self.data_representation+"_"+model_name + "_parametres_"+name+"_accuracy_test_"+str(round(test_score,3))

        cm_fig.ax_.set_title(experiment_name)

        print("Metrics")
        print("train_score ==>",train_score)
        print("validation_score ==>",validation_score)
        print("test_score ==>",test_score)

        print("Save experiment resources.....")
        self.save_experiment(experiment_name,model_name,grid_search,best_parametres,train_score,validation_score,test_score,cm_fig,number_fits)
        print("Experiment executed successfully")
        print("#"*50)