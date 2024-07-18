import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd  
import math

class Metrics:
    def __init__(self,num_of_retrieved,res1=0):
        self.num_of_retrieved=num_of_retrieved          
        self.res1=res1
    def rel_num_list(self):
        rq_path='/home/sarantis/Documents/InformationRetrieval/Project/Relevant_20'         #path
        relevant_docs_for_q=[]                                                              #αριθμός σχετικών κειμένων
        rel_docs=[]                                                                         #σχετικά κείμενα
        with open(rq_path, 'r') as f:                                                       #διάβασμα αρχείου
            for line in f:                                                                  #Για κάθε γραμμή και άρα και query με το split παίρνουμε τα σχετικά κείμενα
                token_num = line.split()                                                    #Κάνουμε append τα κείμενα στο ένα array και στο άλλο τον length της λίστας που πήραμε από το split.
                rel_docs.append(token_num)
                relevant_docs_for_q.append(len(token_num))
        return relevant_docs_for_q,rel_docs
    def find_rel(self,i):
        idc,rd=self.rel_num_list()
        rel = [int(item1) for item1 in self.res1[i] for item2 in rd[i] if int(item1) == int(item2)]        
        return len(rel)
    def recall(self): 
        rdq,idc=self.rel_num_list()
        recall = []
        for i in range(0,20):                                                   #Για κάθε query καλούμε την find_rel για να βρούμε τα σχετικά κείμενα που ανακτήσαμε
            recall_q=self.find_rel(i) / rdq[i]
            recall.append(recall_q)                                             #Υπολογίζουμε το recall για κάθε query και βρίσκουμε τον μέσο όρο τους
        print(f'The recall metric of the model is: {np.mean(recall)} \n\n')
        return np.mean(recall)
    def precision(self):
        precision=[]
        for i in range(0,20):
            precision_q=self.find_rel(i) / self.num_of_retrieved
            precision.append(precision_q)
                
        print(f'The precision metric of the model is: {np.mean(precision)} \n\n')
        return np.mean(precision)  
    def MesiArmonikiTimi(self):
        pre=self.precision()
        rec=self.recall()
        F = ( 2 * rec * pre ) / (pre+rec)
        return F


    def sum_of_digits(self,n):
        
        r=sum(int(digit) for digit in str(abs(int(n))))             #Υπολογισμός βαθμών 
        if r<2:
            return 0
        elif r>=2 and r<4:
            return 1
        elif r>=4:
            return 2
    def get_res_list(self):
        result_list=[]
        with open('/home/sarantis/Documents/InformationRetrieval/Project/cfquery_detailed', 'r') as file:
            values = []  
            flag = 0

            for line in file:                                                           #Διάβασμα του αρχείου
                parts = line.split()                                                

                if parts and parts[0] == 'RD':                                          #Όταν η γραμμή ξεκινάει με RD κρατάμε τιμές μέχρι να βρούμε την άδεια γραμμή (not parts)
                    flag = 1
                    values = []  

                    for i in range(1, len(parts)):
                        if i % 2 == 0:
                            values.append(self.sum_of_digits(parts[i]))
                        else:
                            values.append(int(parts[i]))

                    result_list.append(values)
                elif not parts:
                    flag = 0
                    values = []  
                elif flag == 1:
                    for i in range(len(parts)):
                        if i % 2 == 1:
                            values.append(self.sum_of_digits(parts[i]))
                        else:
                            values.append(int(parts[i]))
        return result_list
    def DCG(self):
        final = []
        append_next=0
        for j in range(20):                                                                       #Για κάθε query
            comp = [] 
            for num in self.res1[j]:                                                              #Για κάθε κείμενο που κάναμε retrieve
                found = 0
                for index,num2 in enumerate(self.get_res_list()[j]):                              #Ψάξτο στην λίστα του προηγούμενο function αν το βρεις στις ζυγές θέσεις κράτα το βαθμό σχετικότητας της επόνενης θέσης
                    if index % 2 == 0 or append_next==1:
                        if append_next==1 :
                            comp.append(num2)
                            append_next=0
                        if num == num2:
                            append_next=1
                            found=1
                    if found == 0 and index == len(self.get_res_list()[j]) -1:

                        comp.append(0)
                if len(comp)==self.num_of_retrieved:
                    final.append(comp)
        sortedfinal=[]
        for item in final:
            s=sorted(item)
            s.reverse()
            sortedfinal.append(s)
        return final,sortedfinal
    def create_gvec(self,vec):
        for item in vec:
            sumi=0
            for i in range(len(item)):
                if i!=0:
                    item[i] = item[i] / np.log2(i+1)    #discount
                    item[i]=item[i]+sumi                #cumulation
                    item[i] = round(item[i], 2)
                sumi=item[i]
        ml=[]
        for i in range(len(vec[0])):
            test=0
            for j in range(len(vec)):
                test+=vec[j][i]
            test_f= test / len(vec)
            test_f=round(test_f,2)
            ml.append(test)
        return ml
    def calculate_ndcg(self):
        dcg , ndcg = self.DCG()
        dcg= self.create_gvec(dcg) 
        ndcg= self.create_gvec(ndcg)
        res = []
        for i in range(len(dcg)):
            res.append(dcg[i] / ndcg[i])
        return res

class VSM:
    def __init__(self,df):
        self.df=df
    def tf(self,df): 
       tf_df=pd.DataFrame()
       tf_df=df
       tf_df=tf_df.div(tf_df.max(axis=1),axis=0)                        #Πάρε το max κάθε γραμμής και διαίρεσε όλη την γραμμή με αυτό
       return tf_df
    def idf(self):
        idf_arr=[]
        idf_arr=np.count_nonzero(self.df, axis=0)                       #Βρες τα μη μεδενικά κελιά κάθε column για κάθε term
        idf_arr=idf_arr.astype(float)
        for index, value in enumerate(idf_arr):
            if value == 0:
                continue
            else:
                idf_arr[index] = math.log10(len(self.df.index) / value)
        return idf_arr
    def idf_to_df(self):     
        idf_df=pd.DataFrame(data=self.idf().reshape(1, -1),columns=self.df.columns)
        return idf_df
    def calculate_W(self,df): 
        W=self.tf(df).mul(self.idf(),axis=1)
        
        return W
    def file_DTW(self,df): 
        
        return self.calculate_W(df).apply(lambda row: row**2, axis=1)
    def paranomastis(self):
        paranomastis=self.file_DTW(self.df).sum(axis=1)
        paranomastis=np.sqrt(paranomastis)
        return paranomastis
    def weight(self):
        W = self.calculate_W(self.df)
        result_df = self.calculate_W(self.df).div(self.paranomastis().values, axis=0)
        return result_df
    def get_query(self,n):  
        questions = []
        with open("/home/sarantis/Documents/InformationRetrieval/Project/Queries_20", 'r') as file:
            lines = file.readlines()
            i = 0

            while i < len(lines):
                questions.append(lines[i].strip())
                i += 1

                vsm_common = self.idf_to_df()

        query1=questions[n].split()
        #query1=self.filter_w(query1)
        data_q = {}
        for token in query1:
            if token in data_q:
                data_q[token] += 1
            else:
                data_q[token] = 1


        query_pd=pd.DataFrame(data_q,index=["0"])
        df_query_big = pd.DataFrame(np.zeros((1, 11368)),columns=vsm_common.columns) 
        query_pd.columns = query_pd.columns.str.upper() 

        for col in query_pd.columns:
            df_query_big[col] = query_pd[col].values

        return df_query_big
    def query_res(self,n):  
        df_query=self.get_query(n)
        res_df= self.tf(df_query)

        res_df=0.5*res_df
        for col in res_df.columns:
            res_df[col] = res_df[col].apply(lambda x: x + 0.5 if x > 0 else 0)
        idf=self.idf_to_df()

        query_res= idf  * res_df 
        return query_res
    def term_A(self,n):
        doc_W = self.weight()
        term_a = doc_W * self.query_res(n).iloc[0]
        term_as=term_a.sum(axis=1)
        return term_as
    def term_B(self,n):
        DocW2=self.file_DTW(self.weight())
        DocW_s=DocW2.sum(axis=1)
        query_tb=self.query_res(n).apply(lambda row: row**2, axis=1)
        query_tb1=query_tb.sum(axis=1)

        term_b=np.sqrt(DocW_s * query_tb1.loc[0])  
        return term_b
    def Calc_Weights(self,n):
        res=self.term_A(n) / self.term_B(n)

        ten_largest_values = pd.Series(res).nlargest(10)
        indices_of_ten_largest_values = ten_largest_values.index
        print("Three largest values:\n", ten_largest_values)
        print("Indices of three largest values:", indices_of_ten_largest_values)
        return indices_of_ten_largest_values
   ######################################################################################     
    def Weighted_create(self):
        Dtw2=self.tf(self.df)
        Dtcw=0.5*Dtw2
        for col in Dtcw.columns:
            Dtcw[col] = Dtcw[col].apply(lambda x: x + 0.5 if x > 0 else 0)
        return Dtcw
    def safe_log10(self,value):
        if value > 0 and value<len(self.df.index):
            res=math.log10((len(self.df.index) - value) / value)
            if res>=0:
                return res
            else:
                return 0
        else:
            return 0
    def weighted_query(self,n):
        query_v2=self.get_query(n)                          #Παίρνουμε το query
        test = self.df * query_v2.iloc[0]                   #Δημιουργούμε το test df που είναι στην ουσία το αρχικό df αλλά μόνο με τους όρους που υπάρχουν στο query
        test = test.apply(lambda x: (x !=0).sum())          #Μέτρα όλα τα μη μηδενικά στοιχεία 
        test = test.apply(self.safe_log10)                  #Apply την παραπάνω συνάρτηση
        test_df=test.to_frame()                             #Ήταν series και το κάναμε dataframe
        test_df=test_df.transpose()                         
        return test_df
    def term_Av2(self,n):
        term_a_v2=self.Weighted_create() * self.weighted_query(n).iloc[0]
        term_a_v2=term_a_v2.sum(axis=1)
        
        return term_a_v2
    def term_Bv2(self,n):
        Dtw2_termb=self.Weighted_create().apply(lambda row: row**2, axis=1)
        quert_v2_termb=self.weighted_query(n).apply(lambda row: row**2, axis=1)
        Dtw2_termb=Dtw2_termb.sum(axis=1)
        quert_v2_termb=quert_v2_termb.sum(axis=1)
        term_b_v2=np.sqrt(Dtw2_termb * quert_v2_termb.loc[0])
        return term_b_v2
    def propabilistic_res(self,n):
        res_2=self.term_Av2(n) / self.term_Bv2(n)


        ten_largest_values_2 = pd.Series(res_2).nlargest(10)
        indices_of_ten_largest_values_2 = ten_largest_values_2.index

        print("Three largest values:\n", ten_largest_values_2)
        print("Indices of three largest values:", indices_of_ten_largest_values_2)
        return indices_of_ten_largest_values_2
    def filter_w(self,array):
        filter_words = ["ON","IN","FROM","THE","OF","WHAT","WHERE","THIS"]
        array = [word.upper() for word in array if word.upper() not in filter_words]
        return array
    def get_res(self):
        results1=[]
        results2=[]
        for res in range(0,20):
            results2.append(self.propabilistic_res(res).tolist())
            results1.append(self.Calc_Weights(res).tolist())
        results1 = [[int(element) for element in sublist] for sublist in results1]
        results2 = [[int(element) for element in sublist] for sublist in results2]
        # for item in results1:
        #     item.reverse()
        return results1,results2

