from gurobipy import *
import numpy as np
import pandas as pd
import os

# =============================================================================
# 데이터 import 및 전처리
# =============================================================================
os.getcwd()
os.chdir('C:/python/gurobi/data_for_gurobi')

project=pd.read_csv('select_best10.csv')
project=np.asarray(project)
project = project[:, 1:project.shape[1]]

sum_of_items = project[:,0]
items = project[:,1:project.shape[1]]


# =============================================================================
# step1. 의사결정변수 설계
# =============================================================================
m_project = Model("project")

select_k = 10

num_of_orders = items.shape[0]
num_of_items = items.shape[1]

x1 = []
for i in range(num_of_items):
    x1.append(m_project.addVar(vtype=GRB.BINARY))
    
x2 = []
for i in range(num_of_orders):
    x2.append(m_project.addVar(vtype=GRB.BINARY))



# =============================================================================
# step2. 제약식설정
# =============================================================================

# 제약식1. 한 작업자당 10개씩의 상품만 배정
contr1 = m_project.addConstr( np.sum(x1) == 10 ) 

# 제약식2. 이진변수로 설정한 x2가 1이 되면 해당 주문은 만족되는 것으로 설정
contr2 = m_project.addConstrs( (np.dot(items[i,:],x1)/sum_of_items[i]) >= x2[i] for i in range(num_of_orders) ) 




# =============================================================================
# step3. 목적함수 설정
# =============================================================================
m_project.setObjective(np.sum(x2), GRB.MAXIMIZE)
m_project.update()


# =============================================================================
# step4. 최적화
# =============================================================================
m_project.optimize()

# 한명의 작업자가 10개의 아이템만을 처리하였을 때, 처리가능한 최대(최적)주문의 수
m_project.objval



# =============================================================================
# step4. 결과정리
# =============================================================================
result = np.empty((num_of_items,1))

for i in range(num_of_items):
    result[i,0] = x1[i].getAttr(GRB.Attr.X)

#해당 column index에 해당하는 아이템 10개를 작업자에게 배정하는 것이 최적해이다.    
print('\nBest_10')
print(np.nonzero(result)[0])

