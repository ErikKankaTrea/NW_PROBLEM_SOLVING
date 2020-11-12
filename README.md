## NW_PROBLEM_SOLVING (test)

NOTE: Before testing it - You should: </p>
1. Have installed the libraries pointed in the requirements.txt file.</p>
2. Download german word2Vec file from [here](https://devmount.github.io/GermanWordEmbeddings/) - scrolling down you can find "Download" section.</p> 


You will find the following structure of files:</p>
. /New Work
   - /data  [Contains raw data] // YOU HAVE TO ADD HERE the german word2Vec model by its default name "german.model"
   - /notebooks [Analysis on data in order to set two/three approaches that have been tried]
   - /output [Outputs of preprocessed data and the matching tables 1:n]
   - /rest_api [Easy/Dirty/Quick api with flask]
   - /src [utilities file with functions, common paths decorator functions, and the prototypes]
   - settings [you add your input/output paths - model and data share same input path]

</p>


For the Flask API only two pages http://0.0.0.0:8000 (welcome page) http://0.0.0.0:8000/apidocs/ (api page) [change the 0.0.0.0 for your IPv4]


Addiotional comments: 
Beyond the poor results, I agree that there is a large-finite ways of doing it, also adding their combinations. 
["fuzzy join", "matrix factorization 10k X 50K", combine first approach then second approach ] 

Cheers!!
