### Preparing the dataset for TREC

We have two sets of json files. one (TREC-IS-Client-v3.jar - all json files containing tweets) where we will extract the tweets (more information can be found here : http://dcs.gla.ac.uk/~richardm/TREC_IS/2019/2019_Instructions.html) and 
the other - "TRECIS-2018-TestEvents-Labels" - for extracting the labels or categories of the tweets. (more info here: http://dcs.gla.ac.uk/~richardm/TREC_IS/
under Training Resources - 2018 Labels for Training (ZIP)

The structure of the json files where the tweets are present is shown below. 

```
annotators are just the wiki information about the events
events - there are 6 events in total - each has the following
 
 events 
     |_event1
     |      |_ eventid
     |      |_  tweets   
     |             |_[{tweet_1: id, category et al}, {tweet_2}, {tweet_2},......{tweet_n}]
     |
     |_ event2
     |       |_ eventid
     |       |_  tweets   
     |             |_[{tweet_1: id, category et al}, {tweet_2}, {tweet_2},......{tweet_n}]


```

Using the above structure, we parse the json to extract the tweets. This is what the notebook "FeatureLabelExtractor.ipynb" does. Also, 
it is to be noted that training data (i.e., tweets) and labels (categories) are in two different json files, therefore, we create a dataframe
ffrom the label json set with columns = ['tweetids', 'alltopics', 'categories'] and also, we create one more dataframe 
from the training json set with columns = ['tweetids', 'topics', 'tweets']. since, we have two dataframe, we merge the two based on the
'tweetids' column. 

This is how the Features i.e., tweets and labels are extracted for training dataset. 

There's one more file, "FeatureLabelExtractor-test.ipynb" for feature label extracting process however, it is for the test set. The changes 
in this notebook are that, since we don't have any labels for the test set. we just extract the dataset using the columns =  ['tweetids', 'topics', 'tweets']
and write the same to a csv file - test dataset.

### WordVector Representation of the Tweets

we will be using the GloVe pretrained word vector for creating the word vector representaiton of the tweets. In particualr, we use this 'glove.twitter.27B/glove.twitter.27B.100d.txt. 
 
All the csv files that are obtained for each event in the above data preparation process can be access by the code in the notebook:
"WordtoVector_lightSVM_train.ipynb". here all the csv files are in "Features" folder so to use the code as is. Otherwise, just replace the folder name with your location where the files are present and all the results - i.e., the word vector repsentation of the sentence are written to text files and are inturn written in a new folder named "datasets".
 
We now have the final dataset in libSVM format ready for training.

For example: each text file represents one class/category of one event with all the related tweets are in a libSVM format word vector representaion. 

Now we need to implement GADGET SVM. please read here: https://github.com/nitinnat/GADGET. ( The way we use the GADGET SVM is 
mentioned below ).

GADGET SVM, as mentioned in the above github repo, is for one dataset. However, TREC 2019-A has 7 events but in effect there are 15 events. As an example, earthquake is an event, however, files 'guatemalaEarthquake2012', 'nepalEarthquake2015', 'italyEarthquakes2012' anf 'chileEarthquake2014' are three different events under earthquake event. In addition, each event is a seperate dataset for us. As such, we build independent models for each of the events. 

It must also be noted that, GADGET SVM is a linear binary classifier. The TREC 2019-A is a multi-class multi-label classification problem. However, the distribution of the classes is not great so instead of considering this as a multi-class problem, we do binary classifier on each class. In other words, for a given tweet, the class/category under consideration is given label '1' and rest all the  categories are given '-1'. Therefore, on a high level, we have, for each event, we build a binary classifier for each category. 

To use the GADGET SVM for building (# total no.of events * # total no.of categories) models. we need the datasets in a particular format as mentioned in the instructions on the GADGET SVM repo. The whole process is automated and the bashscripts
can be used to create the folder and datasets. 

### Bash SCripts:
#### For Data Preparation

the 'datasets' folder which is obtained after Wordvector representation steps, is to be copied into the 'data' folder in 'GADGET/GADGET/data'. [An example folder for one event and for one category in that event is already present in this repo]

```
bash GADGET/GADGET/data/commands.sh // This will create a directory for each category within in a event and move all the text files of the realted to this category's directory. 

bash GADGET/GADGET/scripts/split_data.sh 
bash GADGET/GADGET/scripts/one_dp.sh

```
Optionally if you want to remove the .dat/csv/txt files that are no longer needed in the training process the GADGET SVM. you can use the below bash scripts. 

```
bash GADGET/remove_datt.sh 
bash GADGET/remove_text.sh 
bash GADGET/GADGET/data/removefolder.sh 
```
#### Running the code
```
bash GADGET/GADGET/bash_train_all.sh
 ```
