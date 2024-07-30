import os
import csv

def search_values_in_files(directory, search_phrase, output_csv):
    results = []

    # Iterate through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Process only text files
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                kvalues=[]
                svalues=[]
                bvalues=[]
                KnapSackDoIt=SFCProcessorMapDoIt=BruteForceDoIt=buckets=boxes=""
                for line in lines:
                    for search_word in search_phrase:
                        if search_word in line:
                            if(no_of_buckets==search_word):
                                buckets = line.split(search_word)[-1].strip()
                            if(no_of_boxes==search_word):
                                boxes = line.split(search_word)[-1].strip()
                                # results.append(filename)
                                # results.append(value)
                            # Extract the value after the search phrase
                            if search_word=="KnapSackDoIt()":
                                # values=[]
                                value1 = line.split(search_word)[1].strip()
                                print(value1[3:13])
                                
                                # if len(values)==2:
                                kvalues.append(value1[3:13])
                                # print(len(values))
                                if(len(kvalues)==2):
                                    KnapSackDoIt=kvalues[1]
                                    # results.append([filename]+ values[1:]) 
                                    kvalues=[] 
                                     
                            if search_word=="BruteForceDoIt()":
                                
                                bvalue1 = line.split(search_word)[1].strip()
                                # print("BruteForce",bvalue1[3:13])
                                
                                # if len(values)==2:
                                bvalues.append(bvalue1[3:13])
                                # print("bbb",bvalues)
                                # print("bvalues",len(bvalues),bvalues)
                                if(len(bvalues)==2):
                                    BruteForceDoIt=bvalues[1]
                                    # print("brue",bvalues,BruteForceDoIt)
                                    # results.append([filename]+ values[1:]) 
                                    bvalues=[]
                            if search_word=="SFCProcessorMapDoIt()":
                                
                                value1 = line.split(search_word)[1].strip()
                                print(value1[3:13])
                                
                                # if len(values)==2:
                                svalues.append(value1[3:13])
                                # print(len(values))
                                if(len(svalues)==2):
                                    SFCProcessorMapDoIt=svalues[1]
                                    # results.append([filename]+ values[1:]) 
                                    svalues=[]
        results.append([filename,buckets,boxes, KnapSackDoIt,BruteForceDoIt,SFCProcessorMapDoIt])
                                      
                        
                        
                    # if search_phrase1 in line:
                    #     # Extract the value after the search phrase
                    #     value1 = line.split(search_phrase1)[-1].strip()
                    #     # results.append([filename, value])
                    # if search_phrase2 in line:
                    #     # Extract the value after the search phrase
                    #     value2 = line.split(search_phrase2)[-1].strip()
                    #     results.append([filename, value1,value2])
    print(output_csv)
    # Write results to a CSV file
    with open(output_csv, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Filename', 'buckets','boxes','KnapSackDoIt','BruteForceDoIt','SFCProcessorMapDoIt'])
        csv_writer.writerows(results)

# Set the directory containing the text files
directory_path = '/global/homes/k/kamal/amrex-tests/Graphs/LB/output'
# Set the phrase to search for
no_of_buckets = 'Number of buckets:'
# Set the phrase to search for
no_of_boxes = 'Number of boxes:'
# Set the phrase to search for
search_phrase3 = 'KnapSackDoIt()'
search_phrase=['Number of buckets:','Number of boxes:','KnapSackDoIt()','BruteForceDoIt()','SFCProcessorMapDoIt()']
# Set the output CSV file path
output_csv_path = '/global/homes/k/kamal/amrex-tests/Graphs/LB/output.csv'

# Call the function
search_values_in_files(directory_path, search_phrase,  output_csv_path)
