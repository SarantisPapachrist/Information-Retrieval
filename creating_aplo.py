import os 
import pandas as pd  

def creating_df():
    path = "/home/sarantis/Documents/InformationRetrieval/Project/docs"
    os.chdir(path) 

    simple_dictionary={}                                ##απλο dictionary που λεει ποσες φορες εμφανιστηκε και σε ποιο έγγραφο η καθε λεξη (οχι γραμμες)


    def read_text_file(file_path): 
    
        with open(file_path, 'r') as f:
            tokens=[]
            showed_up={}
            times={}
            f_name=''
            for line in f:                                                  ##διαβαζουμε το αρχειο        
                f_name=os.path.basename(f.name)            
                check_token = line.strip()                                  ##δημιουργουμαι τα τοκενς
                tokens.append(check_token)                                  ##τα βαζουμε σε ενα array
            for index,token in enumerate(tokens):                           ##κανουμε ιτερειτ στο array 
                if token in showed_up:                                      ##αν εχουμε ξαναδιαβάσει αυτο το τοκεν κανουμε το times+1 και βαζουμε στο showup σε ποια γραμμη
                    showed_up[token].append(index+1)        
                    times[token]+=1
                else:                                                       ##αν δεν υπάρχει κανω 1 το times και βαζω στο showed up την γραμμη
                    showed_up[token] = [index+1]
                    times[token] = 1
                    
                if token in simple_dictionary:                              ##Tο ιδιο για το simple dictionary που δεν εχει μεσα τις γραμμες
                    simple_dictionary[token][0]+=1
                    simple_dictionary[token].append((f_name,times[token]))
                else:
                    simple_dictionary[token]=[1,(f_name,times[token])]
                    
                
    for file in os.listdir(): 
        file_path = os.path.join(path, file)  
        
        read_text_file(file_path)
        

    data_dict = {}
    doc_ids = []                                                            # Λίστα για τα document IDs

    for data in simple_dictionary.values():
        for item in data[1:]:
            doc_ids.append(item[0])                                         # Βάζουμε τα ID των αρχείων στη λίστα 

                                                                            # Sort και unique (όχι διπλά) document IDs
    doc_names = sorted(set(doc_ids))

    for token in simple_dictionary:
        data_dict[token] = {doc_id: 0 for doc_id in doc_names}


    for word, word_data in simple_dictionary.items():
        for doc_id, count in word_data[1:]:
            data_dict[word][doc_id] = count

    df = pd.DataFrame(data_dict)
    df.index.name = 'Document'

    return df




