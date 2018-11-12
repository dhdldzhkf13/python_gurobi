from gurobipy import *
import numpy as np
import pandas as pd
import os

# =============================================================================
# 데이터 import 및 전처리
# =============================================================================
os.getcwd()
os.chdir('C:/python/gurobi/data_for_gurobi')

project=pd.read_csv('order2vec_binary_mip.csv')
project=np.asarray(project)
project = project[:, 1:project.shape[1]]

#행 = 거래처, 열 = 아이템종류
project.shape

# =============================================================================
# 데이터 import 완료 및 전처리 완료
# =============================================================================



# =============================================================================
# step1. 의사결정변수 설계
# =============================================================================
m_project = Model("project")

num_k = 5
num_client = project.shape[0]
num_item = project.shape[1]


'''
이런식으로 아래의 637by5행렬 생성 가능
combi_variable = []
for i in range(num_client*num_k):
    combi_variable.append( m_project.addVar(vtype=GRB.BINARY))
combi_variable = np.asarray(combi_variable)
combi_variable = combi_variable.reshape(num_client,5)
combi_variable = np.array(combi_variable)
combi_variable.shape 
type(combi_variable) 
max_constr = m_project.addVar(name='max_constr')
m_project.update()
'''


#637by1 X_Variable produce
combi_variable1 = []
for i in range(num_client):
    combi_variable1.append( m_project.addVar(vtype=GRB.BINARY))
combi_variable1 = np.asarray(combi_variable1)
combi_variable1=combi_variable1.reshape(num_client,1)
combi_variable1.shape

combi_variable2 = []
for i in range(num_client):
    combi_variable2.append( m_project.addVar(vtype=GRB.BINARY))
combi_variable2 = np.asarray(combi_variable2)    
combi_variable2 = combi_variable2.reshape(num_client,1)
combi_variable2.shape
    
combi_variable3 = []
for i in range(num_client):
    combi_variable3.append( m_project.addVar(vtype=GRB.BINARY))
combi_variable3 = np.asarray(combi_variable3)   
combi_variable3 = combi_variable3.reshape(num_client,1)
combi_variable3.shape 
    
combi_variable4 = []
for i in range(num_client):
    combi_variable4.append( m_project.addVar(vtype=GRB.BINARY))
combi_variable4 = np.asarray(combi_variable4)   
combi_variable4 = combi_variable4.reshape(num_client,1)
combi_variable4.shape     

combi_variable5 = []
for i in range(num_client):
    combi_variable5.append( m_project.addVar(vtype=GRB.BINARY))
combi_variable5 = np.asarray(combi_variable5)   
combi_variable5 = combi_variable5.reshape(num_client,1)
combi_variable5.shape 

#637by5 matrix X_Variable complete
combi_variable = np.hstack((combi_variable1,
                            combi_variable2,
                            combi_variable3,
                            combi_variable4,
                            combi_variable5))
combi_variable = np.array(combi_variable)
combi_variable.shape
type(combi_variable)


#max선형화를 위한 의사결정변수
max_constr = m_project.addVar(name='max_constr')







# =============================================================================
# 새로운 의사결정변수와 big_M
# =============================================================================

bigM_variable1 = []
for i in range(num_item):
    bigM_variable1.append( m_project.addVar(vtype=GRB.BINARY))
bigM_variable1 = np.asarray(bigM_variable1)
bigM_variable1 = bigM_variable1.reshape(num_item,1)
bigM_variable1.shape


bigM_variable2 = []
for i in range(num_item):
    bigM_variable2.append( m_project.addVar(vtype=GRB.BINARY))
bigM_variable2 = np.asarray(bigM_variable2)
bigM_variable2 = bigM_variable2.reshape(num_item,1)
bigM_variable2.shape


bigM_variable3 = []
for i in range(num_item):
    bigM_variable3.append( m_project.addVar(vtype=GRB.BINARY))
bigM_variable3 = np.asarray(bigM_variable3)
bigM_variable3 = bigM_variable3.reshape(num_item,1)
bigM_variable3.shape


bigM_variable4 = []
for i in range(num_item):
    bigM_variable4.append( m_project.addVar(vtype=GRB.BINARY))
bigM_variable4 = np.asarray(bigM_variable4)
bigM_variable4 = bigM_variable4.reshape(num_item,1)
bigM_variable4.shape


bigM_variable5 = []
for i in range(num_item):
    bigM_variable5.append( m_project.addVar(vtype=GRB.BINARY))
bigM_variable5 = np.asarray(bigM_variable5)
bigM_variable5 = bigM_variable5.reshape(num_item,1)
bigM_variable5.shape



#5by158 matrix X_Variable complete
bigM_variable = np.hstack((bigM_variable1,
                           bigM_variable2,
                           bigM_variable3,
                           bigM_variable4,
                           bigM_variable5))
bigM_variable = np.array(bigM_variable)
bigM_variable = bigM_variable.T
bigM_variable.shape
type(bigM_variable)

m_project.update()

#여기까지 새로운 의사결정변수 158by5 생성 완료
#이제 big_M계산
big_M = 1000000

bigM_variable_dot_BigM = big_M*bigM_variable
bigM_variable_dot_BigM.shape




# =============================================================================
# step2. 제약식설정
# =============================================================================

#제약식1. 각 거래처는 반드시 하나의 클러스터에만 속하게됨을 뜻하는 제약식을 설정
const1 = m_project.addConstrs(( np.sum( combi_variable[i:i+1,:] ) == 1 for i in range(num_client) ) , name='1')
combi_variable[0:1,:]

m_project.update()




#제약식2. P <= bigM_variable_dot_BigM, 
project.T.shape
combi_variable.shape
const2_Lin_Exp=np.dot(project.T , combi_variable ) # 이 코드를 통해 각 클러스터의 아이템당 시간(10초)를 계산.
const2_Lin_Exp.shape



const2_Lin_Exp.T.shape
bigM_variable_dot_BigM.shape

const2  = m_project.addConstrs((const2_Lin_Exp.T[i,j] <= bigM_variable_dot_BigM[i,j] for i in range(num_k)
                                                                                for j in range(num_item)), name='c')

m_project.update() # 여기에 제약식개수가 아래의 식에서 나오는 수와 같은지 확인.
num_k*num_item+num_client



#제약식3.  Max함수를 선형화 하기위해서 Max_constr을 사용하는 코드.

assigne_time = 10
fixed_time = 60

type(np.sum(const2_Lin_Exp[:,0:1])*assigne_time)
type(np.sum(bigM_variable.T[:,0:1])*fixed_time)
type(np.add(np.sum(const2_Lin_Exp[:,0:1])*assigne_time, np.sum(bigM_variable.T[:,0:1])*fixed_time))
            
                            
const3_01 = m_project.addConstr( np.add(np.sum(const2_Lin_Exp[:,0:1])*assigne_time, np.sum(bigM_variable.T[:,0:1])*fixed_time) <= max_constr )# 이 코드를 통해 각 클러스터의 아이템당 시간(10초)를 계산.
const3_02 = m_project.addConstr( np.add(np.sum(const2_Lin_Exp[:,1:2])*assigne_time, np.sum(bigM_variable.T[:,1:2])*fixed_time) <= max_constr )
const3_03 = m_project.addConstr( np.add(np.sum(const2_Lin_Exp[:,2:3])*assigne_time, np.sum(bigM_variable.T[:,2:3])*fixed_time) <= max_constr )
const3_04 = m_project.addConstr( np.add(np.sum(const2_Lin_Exp[:,3:4])*assigne_time, np.sum(bigM_variable.T[:,3:4])*fixed_time) <= max_constr )
const3_05 = m_project.addConstr( np.add(np.sum(const2_Lin_Exp[:,4:5])*assigne_time, np.sum(bigM_variable.T[:,4:5])*fixed_time) <= max_constr )



#제약식4, 클러스터당 150개씩 제약설정
limit_cluster = 150

#const4 = m_project.addConstrs(( np.sum( combi_variable[:,i:i+1] ) <= limit_cluster for i in range(num_k) ) , name='4')


m_project.update()


# =============================================================================
# step3. 목적함수 설정
# =============================================================================
m_project.setObjective(max_constr, GRB.MINIMIZE)
m_project.update()

# =============================================================================
# step4. 실행 및 확인
# =============================================================================
time_limit = 100;

m_project.setParam(GRB.Param.TimeLimit, time_limit)

m_project.optimize()

m_project.getVars()
m_project.objval

m_project




# =============================================================================
# step6. 결과 정리
# =============================================================================

#우선 각 거래처들이 어떤 클러스터에 배정되었는지 확인할수 있는 결과 variable을 추출
result = np.empty((num_client,num_k))

for i in range(num_k):
    for j in range(num_client):
        result[j,i] = combi_variable[j,i].getAttr(GRB.Attr.X)


#각 클러스터마다의 거래처 개수 계산
cluster_num = np.sum(result,axis=0)


#원데이터세트(637, 158)과 결과variable(637, 5)을 np.dot하여 Linear_Exp, 즉 P를 계산
project.shape
result.shape

P = np.dot(project.T,result).T


#이제 이 P를 이용하여 각 클러스터의 소요시간을 계산.
#10초
time_10 = np.sum(P, axis=1)*assigne_time

#60초
time_60 = np.sum(((P>=1)*fixed_time),axis=1)

#10+60초
time_total = np.add(time_10, time_60)



# =============================================================================
# 최종결과
# =============================================================================
print("\n\n\n### Result! Assigned Clusters ###\n")
for i in range(num_k):
    print("Clust%s_assignClient : %s" % (i+1,cluster_num[i]))
    
print()  
for i in range(num_k):
   print("Clust%s_totalTime : %s" % (i+1,time_total[i]))   

print()
print('mean : %s' % np.mean(time_total) )
print('Max-Min : %s' % (np.max(time_total)-np.min(time_total)) )



# =============================================================================
# step5. 결과 엑셀로 
# =============================================================================
pd.DataFrame(result).to_csv('result_test_client2.csv')




'''
NP-난해문제이므로
최적해를 구하기는 어렵지만
gurobi에서 제공하는
time_limit파라미터를 이용하여
적절한 시간내에 좋은 가능해를 구할 수 있다.
'''
