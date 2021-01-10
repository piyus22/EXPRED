Objective- Development of machine-learning based approaches for the identification of expansin proteins

Why Expansins?
Expansins play an important role in cell wall remodeling and disassembly necessary for plant growth and acclimatization[1], apart from this also plays a crucial role in activities such as differentiation, transport, communication, senescence, abscission, plant-pathogen interactions and ultimately plant growth[2]

Enable cell expansion by triggering a pH dependent relaxation of the cell wall which loosens and softens the cell wall.

Areas of application  :-
-Increased fruit firmness ( To increase the shelf life of fruits and vegetables)
-Increased rate of light-induced stomatal opening and reduced sensitivity of stomata to the stimuli [3]
-Enhanced growth and larger leaves under normal growth conditions[4]
-Accelerated root growth[6]

Steps followed to build a Machine learning (ML) model :-
1) Data Collected
2) Redundancy removed
3) Dataset in 2 ratios (1:1 and 1:2)
3) Feature calculation
4) Different ML models were built (ex- SVM, Decision Tree Classifier e.t.c)
5) Finally two features in two different ratios and their corresponding model were choosen (SVM with rbf kernel)
6) Flask was used to create the webserver

LINK TO ACCESS THE SEVER - http://expred.pythonanywhere.com/

Python Libraries required :-
-NumPy
-scikit-learn
-Biopython
-Flask
-Matplotlib
-PyCaret
-Pandas


PURPOSEFUL OMISSION:-
1) Sequence data files
2) Corresponding feature scores (csv files)



References
1.Kong, Youbin, et al. "GmEXLB1, a soybean expansin-like B gene, alters root architecture to improve phosphorus acquisition in Arabidopsis." Frontiers in plant science 10 (2019).
2.Fukuda H (ed) (2014) Plant cell wall patterning and cell shape. Wiley, Hoboken
3.Chen, Yongkun, et al. "A comprehensive expression analysis of the expansin gene family in potato (Solanum tuberosum) discloses stress-responsive expansin-like B genes for drought and heat tolerances." PloS one 14.7 (2019): e0219837
4.Kende H, Bradford KJ, Brummell Da, Cho HT, Cosgrove DJ, Fleming AJ, Voesenek LACJ (2004) Nomenclature for members of the expansin superfamily of genes and proteins. Plant Mol Biol 55(3):311–314
