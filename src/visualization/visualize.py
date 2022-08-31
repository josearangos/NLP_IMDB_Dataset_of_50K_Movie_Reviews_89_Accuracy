from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import learning_curve

class Visualizer:
    """
    This class allows to visualize some charts

    Args:
        df_data (PandasDataFrame): Pandas dataFrame with clean data without stemming process
        max_words (int, optional): max number of words include in the word cloud chart. Defaults to 300.


    Methods: 
        convert_array_to_text(self,sentiment:int)
        This method convert an array of str in str, join the rows in an only sting 

        plot_word_cloud(self,corpus:list,figsize=(20,20))
            This method retur a figure of word cloud

        plot_frequency_words(self,corpus:list, label:str, top=12, figsize=(10,5))
            This method create a horizontal bar chart with most frequency words


    """


    def __init__(self,df_data,max_words=300):
        """This class allows to visualize some charts

        Args:
            df_data (PandasDataFrame): Pandas dataFrame with clean data without stemming process
            max_words (int, optional): max number of words include in the word cloud chart. Defaults to 300.
        """

        self.word_cloud =WordCloud(width=1100,height=600,max_words=max_words,min_font_size=5) #Word cloud
        self.df_data = df_data


    def convert_array_to_text(self,sentiment:int): 
        """This method convert an array of str in str, join the rows in an only sting 

        Args:
            sentiment (int): sentiment 1:Positive, 0:Negative

        Returns:
            str: Big string with all corpus
        """

        texts = []    
        data = self.df_data[self.df_data["sentiment"] == sentiment]["review"].values
            
        for d in data:
            for text in d.split(" "):
                texts.append(text)
                
        return texts

    def plot_word_cloud(self,corpus:list,figsize=(20,20)):
        """This method retur a figure of word cloud

        Args:
            corpus (list): list of words of the corpus
            figsize (tuple, optional): figure size. Defaults to (20,20).

        Returns:
            Figure : word cloud figure
        """
        plt.figure(figsize=figsize)
        corpus_word_cloud = self.word_cloud.generate(" ".join(corpus))
        return corpus_word_cloud   


    def plot_frequency_words(self,corpus:list, label:str, top=12, figsize=(10,5)):
        """
        This method create a horizontal bar chart with most frequency words

        Args:
            corpus (list): list of words of the corpus
            label (str): description to figure title only take two values "Positive/Negative"
            top (int, optional): Number of words most frequency that it will be plotting. Defaults to 12.
            figsize (tuple, optional): figure size.. Defaults to (10,5).

        Returns:
            Figure: horizontal bar chart
        """
        corpus_counter=Counter(corpus)
        most_frequency_words= corpus_counter.most_common()

        frequency_words = []
        words_most_frequency = []
        for w,f in most_frequency_words[0:top]:
            frequency_words.append(f)
            words_most_frequency.append(w)
            
        fig= plt.figure(figsize=figsize) 
        plt.barh(words_most_frequency[::-1],frequency_words[::-1])
        plt.xlabel("Frequency")
        plt.ylabel("Words")
        plt.title("Frequency "+label+" words")
        
        return fig


    def plot_learning_curve(self,estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        axes : array of 3 axes, optional (default=None)
            Axes to use for plotting the curves.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

            - None, to use the default 5-fold cross-validation,
            - integer, to specify the number of folds.
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like, shape (n_ticks,), dtype float or int
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the dtype is float, it is regarded as a
            fraction of the maximum size of the training set (that is determined
            by the selected validation method), i.e. it has to be within (0, 1].
            Otherwise it is interpreted as absolute sizes of the training sets.
            Note that for classification the number of samples usually have to
            be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True, scoring='f1')
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1,
                            color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                            fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt



    def plot_roc(self,Xtest, Ytest, probs, xlabel):
        ns_probs = [0 for _ in range(len(Ytest))]
        
        probs = probs[:, 1]
        ns_auc = roc_auc_score(Ytest, ns_probs)
        auc = roc_auc_score(Ytest, probs)  

        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Logistic: ROC AUC=%.3f' % (auc))

        ns_fpr, ns_tpr, _ = roc_curve(Ytest, ns_probs)
        fpr, tpr, _ = roc_curve(Ytest, probs)   

        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label= xlabel)

        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()




