import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

cell_df=pd.read_csv('DC_negative_1isto2.csv')
cell_df.columns
feature_df=cell_df[["AA","RA","NA","DA","CA","EA","QA","GA","HA","IA","LA","KA","MA","FA","PA","SA","TA","WA","YA","VA","AR","RR","NR","DR","CR","ER","QR","GR","HR","IR","LR","KR","MR","FR","PR","SR","TR","WR","YR","VR","AN","RN","NN","DN","CN","EN","QN","GN","HN","IN","LN","KN","MN","FN","PN","SN","TN","WN","YN","VN","AD","RD","ND","DD","CD","ED","QD","GD","HD","ID","LD","KD","MD","FD","PD","SD","TD","WD","YD","VD","AC","RC","NC","DC","CC","EC","QC","GC","HC","IC","LC","KC","MC","FC","PC","SC","TC","WC","YC","VC","AE","RE","NE","DE","CE","EE","QE","GE","HE","IE","LE","KE","ME","FE","PE","SE","TE","WE","YE","VE","AQ","RQ","NQ","DQ","CQ","EQ","QQ","GQ","HQ","IQ","LQ","KQ","MQ","FQ","PQ","SQ","TQ","WQ","YQ","VQ","AG","RG","NG","DG","CG","EG","QG","GG","HG","IG","LG","KG","MG","FG","PG","SG","TG","WG","YG","VG","AH","RH","NH","DH","CH","EH","QH","GH","HH","IH","LH","KH","MH","FH","PH","SH","TH","WH","YH","VH","AI","RI","NI","DI","CI","EI","QI","GI","HI","II","LI","KI","MI","FI","PI","SI","TI","WI","YI","VI","AL","RL","NL","DL","CL","EL","QL","GL","HL","IL","LL","KL","ML","FL","PL","SL","TL","WL","YL","VL","AK","RK","NK","DK","CK","EK","QK","GK","HK","IK","LK","KK","MK","FK","PK","SK","TK","WK","YK","VK","AM","RM","NM","DM","CM","EM","QM","GM","HM","IM","LM","KM","MM","FM","PM","SM","TM","WM","YM","VM","AF","RF","NF","DF","CF","EF","QF","GF","HF","IF","LF","KF","MF","FF","PF","SF","TF","WF","YF","VF","AP","RP","NP","DP","CP","EP","QP","GP","HP","IP","LP","KP","MP","FP","PP","SP","TP","WP","YP","VP","AS","RS","NS","DS","CS","ES","QS","GS","HS","IS","LS","KS","MS","FS","PS","SS","TS","WS","YS","VS","AT","RT","NT","DT","CT","ET","QT","GT","HT","IT","LT","KT","MT","FT","PT","ST","TT","WT","YT","VT","AW","RW","NW","DW","CW","EW","QW","GW","HW","IW","LW","KW","MW","FW","PW","SW","TW","WW","YW","VW","AY","RY","NY","DY","CY","EY","QY","GY","HY","IY","LY","KY","MY","FY","PY","SY","TY","WY","YY","VY","AV","RV","NV","DV","CV","EV","QV","GV","HV","IV","LV","KV","MV","FV","PV","SV","TV","WV","YV","VV"]]

x=np.asarray(feature_df)
y=np.asarray(cell_df['class'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=0)
print(type(x_train))
#y_test.shape=51
#x_train.shape=(201,20)
#x_test.shape=51,20
from sklearn import svm
classifier=svm.SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
#classifier=svm.SVC(kernel='linear',gamma='auto', C=2)
classifier.fit(x_train,y_train)
#print(classifier)
y_predict=classifier.predict(x_test)
#print(y_predict)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
from sklearn.metrics import confusion_matrix
#print("Confusion matrix: \n",confusion_matrix(y_test, y_predict))
from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test, y_predict))
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
#mod_ROC=roc_auc_score(y_test,classifier.predict(x_test))
#fpr,tpr, _ =roc_curve(y_train, y_test)
#probs=classifier.predict_proba(x_test)
#print(probs)
#probs=probs[:,1]
#fpr, tpr, _ = roc_curve(y_test, probs)
#fpr, tpr, threshold = roc_curve(y_test, probs)
#plt.plot([0,1],[0,1], linestyle='--')
#plt.plot(fpr,tpr,marker='.')

#print(y_train)


#print(probs)
#roc_auc_score(y_test, probs)
pickle.dump(classifier, open('SVMmodel_DP.pkl','wb'))
model = pickle.load(open('SVMmodel_DP.pkl','rb'))
#x=model.predict([[0.1271186441,0.0169491525,0.0169491525,0.0254237288,0.0423728814,0.0254237288,0.0254237288,0.1101694915,0.0084745763,0.0508474576,0.0593220339,0.0677966102,0.0084745763,0.0593220339,0.0508474576,0.0677966102,0.1101694915,0.0084745763,0.0338983051,0.0847457627]])
#x=model.predict([[0.0716845878,0.0286738351,0.0573476703,0.0501792115,0.0286738351,0.0322580645,0.0215053763,0.1111111111,0.0107526882,0.0609318996,0.0609318996,0.0681003584,0.0394265233,0.0501792115,0.0465949821,0.0322580645,0.082437276,0.0322580645,0.0501792115,0.064516129]])
x=model.predict([[0,0.0037313433,0,0,0.0037313433,0.0037313433,0,0.0037313433,0,0.0037313433,0.0037313433,0.0037313433,0.0037313433,0.0037313433,0,0.0111940299,0,0.0037313433,0.0037313433,0.0111940299,0.0037313433,0.0074626866,0.0037313433,0.0037313433,0.0037313433,0,0,0.0037313433,0,0.0037313433,0.0037313433,0,0,0,0.0037313433,0.0037313433,0,0,0.0037313433,0,0,0.0037313433,0,0,0.0037313433,0,0.0037313433,0,0,0,0.0037313433,0.0037313433,0,0.0037313433,0.0074626866,0.0074626866,0.0037313433,0,0.0037313433,0.0037313433,0.0074626866,0.0037313433,0,0.0037313433,0,0,0,0.0074626866,0,0.0037313433,0,0.0037313433,0,0,0,0.0111940299,0,0,0,0.0037313433,0.0037313433,0.0037313433,0,0,0,0,0,0.0037313433,0.0037313433,0.0037313433,0,0.0037313433,0,0,0,0,0,0.0037313433,0.0037313433,0,0.0037313433,0.0037313433,0,0,0,0,0,0,0,0.0037313433,0,0,0,0.0149253731,0,0,0,0,0,0,0,0,0.0037313433,0,0,0,0,0.0111940299,0,0,0,0.0037313433,0,0,0,0,0.0037313433,0,0,0.0037313433,0.0037313433,0.0037313433,0,0.0111940299,0.0074626866,0.0037313433,0,0.0186567164,0,0,0,0.0037313433,0.0037313433,0.0037313433,0.0037313433,0.0074626866,0.0037313433,0.0037313433,0.0111940299,0.0149253731,0,0,0,0,0,0,0,0.0037313433,0,0,0,0.0037313433,0,0,0,0,0,0.0037313433,0,0,0.0074626866,0.0037313433,0,0,0.0037313433,0,0.0037313433,0.0037313433,0.0037313433,0.0037313433,0.0074626866,0.0074626866,0,0,0.0037313433,0.0037313433,0.0037313433,0,0,0,0,0,0.0037313433,0.0111940299,0,0.0037313433,0,0.0074626866,0,0,0.0149253731,0,0,0.0037313433,0.0074626866,0.0074626866,0.0074626866,0.0037313433,0,0.0111940299,0,0,0.0037313433,0,0,0.0037313433,0,0.0037313433,0,0.0074626866,0.0111940299,0,0,0,0.0037313433,0,0,0.0074626866,0,0.0037313433,0,0,0.0037313433,0,0,0,0,0.0037313433,0,0,0,0,0,0,0.0037313433,0,0,0,0,0,0.0037313433,0.0037313433,0,0,0.0037313433,0,0.0037313433,0,0.0037313433,0,0.0037313433,0.0037313433,0,0,0,0.0037313433,0.0074626866,0,0.0037313433,0,0.0111940299,0,0.0074626866,0.0037313433,0,0,0.0037313433,0,0,0.0037313433,0.0074626866,0,0.0037313433,0,0.0037313433,0,0.0074626866,0,0,0,0.0074626866,0,0,0,0,0.0037313433,0,0.0074626866,0,0.0074626866,0.0111940299,0.0037313433,0.0037313433,0,0.0037313433,0.026119403,0.0149253731,0,0.0074626866,0.0037313433,0.0074626866,0.0037313433,0,0,0,0,0.0074626866,0.0037313433,0,0.0074626866,0.0111940299,0,0,0,0.0074626866,0.0074626866,0,0,0,0.0111940299,0,0,0.0074626866,0.0037313433,0,0.0037313433,0,0.0037313433,0,0.0037313433,0,0.0037313433,0,0,0,0,0,0,0,0,0,0.0037313433,0.0074626866,0,0,0,0.0037313433,0.0111940299,0,0,0,0,0,0.0037313433,0,0.0037313433,0.0037313433,0,0.0074626866,0,0.0037313433,0,0.0074626866,0.0074626866,0.0037313433,0.0037313433,0,0.0074626866,0,0.0037313433,0.0037313433,0,0,0.0074626866,0.0037313433,0.0037313433,0.0111940299,0,0,0.0074626866]])

#print(x)
#if x==0:
    #print("Is an expansin sequence")
#else:
    #print("Is not an expansin sequence")
#from sklearn.metrics import plot_confusion_matrix
#print("Confusion matrix: \n",confusion_matrix(y_test, y_predict))
#cm = confusion_matrix(y_test, y_predict)
#tn, fp, fn, tp = cm.ravel()
#print(cm.ravel())
#plot_confusion_matrix(classifier, x_test, y_test)
#from matplotlib import pyplot as plt
#import matplotlib
#matplotlib.use('Agg')
#plt.savefig('Confusion_DP_1isto2')
